from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

try:
    # Keras 3 / TensorFlow >= 2.16
    from keras.utils import PyDataset as _BaseSequence
except ImportError:
    # TensorFlow < 2.16
    from tensorflow.keras.utils import Sequence as _BaseSequence

logger = logging.getLogger(__name__)


# ======================================================================
# 1. Configuration (validée, reproductible, immuable)
# ======================================================================

@dataclass(frozen=True)
class JEPAMaskingConfig:
    """Configuration du masquage selon l'article I-JEPA.

    Attributes:
        img_size: Taille de l'image en pixels (carrée, H == W).
        patch_size: Taille de chaque patch en pixels. Doit diviser `img_size`.
        target_scale_min: Échelle minimale des blocs cibles (fraction de la
            surface de la grille de patches).
        target_scale_max: Échelle maximale des blocs cibles.
        target_aspect_min: Ratio d'aspect minimal des blocs cibles.
        target_aspect_max: Ratio d'aspect maximal des blocs cibles.
        num_targets: Nombre de blocs cibles (M dans l'article).
        context_scale_min: Échelle minimale du bloc de contexte.
        context_scale_max: Échelle maximale du bloc de contexte.
        context_aspect: Ratio d'aspect fixe du bloc de contexte (1.0 = carré).
        min_context_patches: Nombre minimal de patches garantis dans le
            contexte après retrait des chevauchements avec les cibles.
        max_resample_attempts: Nombre d'essais avant d'activer le repli
            garanti si le contexte échantillonné est trop petit.
    """

    img_size: int = 224
    patch_size: int = 16
    target_scale_min: float = 0.15
    target_scale_max: float = 0.20
    target_aspect_min: float = 0.75
    target_aspect_max: float = 1.5
    num_targets: int = 4
    context_scale_min: float = 0.85
    context_scale_max: float = 1.0
    context_aspect: float = 1.0
    min_context_patches: int = 1
    max_resample_attempts: int = 10

    grid_size: int = field(init=False)
    num_patches: int = field(init=False)

    def __post_init__(self) -> None:
        if self.patch_size <= 0 or self.img_size <= 0:
            raise ValueError("img_size et patch_size doivent être > 0.")
        if self.img_size % self.patch_size != 0:
            raise ValueError(
                f"img_size ({self.img_size}) doit être un multiple de "
                f"patch_size ({self.patch_size})."
            )
        object.__setattr__(self, "grid_size", self.img_size // self.patch_size)
        object.__setattr__(self, "num_patches", self.grid_size ** 2)

        self._check_range("target_scale_min", "target_scale_max", 0.0, 1.0)
        self._check_range("target_aspect_min", "target_aspect_max", 1e-3, None)
        self._check_range("context_scale_min", "context_scale_max", 0.0, 1.0)

        if self.num_targets < 1:
            raise ValueError("num_targets doit être >= 1.")
        if self.context_aspect <= 0:
            raise ValueError("context_aspect doit être > 0.")
        if self.min_context_patches < 0:
            raise ValueError("min_context_patches doit être >= 0.")
        if self.min_context_patches > self.num_patches:
            raise ValueError(
                f"min_context_patches ({self.min_context_patches}) ne peut "
                f"pas dépasser num_patches ({self.num_patches})."
            )
        if self.max_resample_attempts < 1:
            raise ValueError("max_resample_attempts doit être >= 1.")

    def _check_range(self, name_min: str, name_max: str,
                      low: Optional[float], high: Optional[float]) -> None:
        vmin, vmax = getattr(self, name_min), getattr(self, name_max)
        if vmin > vmax:
            raise ValueError(f"{name_min} ({vmin}) doit être <= {name_max} ({vmax}).")
        if low is not None and vmin < low:
            raise ValueError(f"{name_min} doit être >= {low}.")
        if high is not None and vmax > high:
            raise ValueError(f"{name_max} doit être <= {high}.")

    @classmethod
    def standard(cls, img_size: int = 64, patch_size: int = 8) -> "JEPAMaskingConfig":
        """Configuration conforme à l'article I-JEPA."""
        return cls(img_size=img_size, patch_size=patch_size)

    @classmethod
    def harder(cls, img_size: int = 64, patch_size: int = 8) -> "JEPAMaskingConfig":
        """Configuration plus difficile, pour limiter le sur-apprentissage."""
        return cls(
            img_size=img_size, patch_size=patch_size,
            target_scale_min=0.20, target_scale_max=0.35, num_targets=6,
            context_scale_min=0.70, context_scale_max=0.90,
        )

    @classmethod
    def very_hard(cls, img_size: int = 64, patch_size: int = 8) -> "JEPAMaskingConfig":
        """Configuration très difficile."""
        return cls(
            img_size=img_size, patch_size=patch_size,
            target_scale_min=0.30, target_scale_max=0.45, num_targets=8,
            context_scale_min=0.50, context_scale_max=0.70,
        )


def _distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """Génère `n` couleurs RGB (0-255) visuellement distinctes.

    Remplace la liste fixe de 4 couleurs de la version initiale, qui se
    répétait de façon peu lisible dès que `num_targets > 4`.
    """
    cmap_name = "tab10" if n <= 10 else "tab20"
    cmap = plt.colormaps[cmap_name]
    return [tuple(int(c * 255) for c in cmap(i % cmap.N)[:3]) for i in range(n)]


# ======================================================================
# 2. Générateur de masques
# ======================================================================

class JEPAMaskGenerator:
    """Générateur de masques de contexte/cibles pour I-JEPA.

    Produit, pour chaque image d'un batch, les indices de patches formant
    le bloc de contexte et les M blocs cibles, en retirant du contexte tout
    chevauchement avec chaque cible (article, section "Masking strategy").
    """

    def __init__(self, config: JEPAMaskingConfig,
                 rng: Optional[np.random.Generator] = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()

    def _sample_block(self, scale_min: float, scale_max: float,
                       aspect_min: Optional[float] = None,
                       aspect_max: Optional[float] = None,
                       aspect_fixed: Optional[float] = None
                       ) -> Tuple[int, int, int, int]:
        """Échantillonne un bloc rectangulaire `(x, y, largeur, hauteur)` en
        unités de patches, à partir d'une échelle et d'un ratio d'aspect."""
        grid = self.config.grid_size
        scale = self.rng.uniform(scale_min, scale_max)
        area = scale * scale * grid * grid  # surface en nombre de patches

        if aspect_fixed is not None:
            aspect = aspect_fixed
        elif aspect_min is not None and aspect_max is not None:
            aspect = self.rng.uniform(aspect_min, aspect_max)
        else:
            raise ValueError("Fournir soit aspect_fixed, soit aspect_min/aspect_max.")

        w_patches = int(round(math.sqrt(area * aspect)))
        h_patches = int(round(math.sqrt(area / aspect)))

        # Le plafond à `grid - 1` (et non `grid`) est volontaire : il
        # garantit toujours au moins un degré de liberté de placement,
        # même pour un bloc occupant presque toute la grille.
        w_patches = max(1, min(w_patches, grid - 1))
        h_patches = max(1, min(h_patches, grid - 1))

        x = int(self.rng.integers(0, grid - w_patches + 1))
        y = int(self.rng.integers(0, grid - h_patches + 1))
        return x, y, w_patches, h_patches

    def _block_to_indices(self, x: int, y: int, w: int, h: int) -> List[int]:
        grid = self.config.grid_size
        rows = np.arange(y, y + h)
        cols = np.arange(x, x + w)
        return (rows[:, None] * grid + cols[None, :]).ravel().tolist()

    def _sample_context_indices(self, excluded: set) -> List[int]:
        """Échantillonne le bloc de contexte en retirant les patches déjà
        utilisés par les cibles, avec un repli garanti si le tirage
        aléatoire ne laisse pas assez de patches disponibles (bug corrigé :
        la version initiale pouvait renvoyer un contexte vide sans avertir
        personne)."""
        cfg = self.config
        for _ in range(cfg.max_resample_attempts):
            x, y, w, h = self._sample_block(
                cfg.context_scale_min, cfg.context_scale_max,
                aspect_fixed=cfg.context_aspect,
            )
            idx = set(self._block_to_indices(x, y, w, h)) - excluded
            if len(idx) >= cfg.min_context_patches:
                return sorted(idx)

        logger.debug(
            "Contexte trop petit après %d essais, repli sur tous les "
            "patches hors cibles.", cfg.max_resample_attempts,
        )
        fallback = set(range(cfg.num_patches)) - excluded
        if not fallback:
            # Cas extrême : les cibles couvrent toute l'image.
            logger.warning(
                "Les blocs cibles couvrent toute l'image : repli sur un "
                "unique patch de contexte."
            )
            fallback = {0}
        return sorted(fallback)

    def generate_masks(self, batch_size: int
                        ) -> Tuple[List[List[int]], List[List[List[int]]]]:
        """Génère les masques de contexte et de cibles pour un batch.

        Returns:
            context_masks: une liste de `batch_size` listes d'indices de patches.
            target_masks_list: une liste de `batch_size` listes de
                `num_targets` listes d'indices de patches.
        """
        context_masks: List[List[int]] = []
        target_masks_list: List[List[List[int]]] = []

        for _ in range(batch_size):
            target_blocks: List[List[int]] = []
            excluded: set = set()
            for _ in range(self.config.num_targets):
                x, y, w, h = self._sample_block(
                    self.config.target_scale_min, self.config.target_scale_max,
                    self.config.target_aspect_min, self.config.target_aspect_max,
                )
                indices = self._block_to_indices(x, y, w, h)
                target_blocks.append(indices)
                excluded.update(indices)

            context_masks.append(self._sample_context_indices(excluded))
            target_masks_list.append(target_blocks)

        return context_masks, target_masks_list

    @staticmethod
    def _blend_patch(vis: np.ndarray, idx: int, grid: int, patch_size: int,
                      color: Tuple[int, int, int], alpha: float = 0.6) -> None:
        row, col = divmod(idx, grid)
        y1, x1 = row * patch_size, col * patch_size
        y2, x2 = y1 + patch_size, x1 + patch_size
        vis[y1:y2, x1:x2] = vis[y1:y2, x1:x2] * (1 - alpha) + np.array(color) * alpha

    def visualize_masks(self, image: np.ndarray, context_mask: Sequence[int],
                         target_masks: Sequence[Sequence[int]]) -> np.ndarray:
        """Superpose les masques de contexte (cyan) et de cibles (couleurs
        distinctes) sur une image, pour inspection visuelle."""
        grid = self.config.grid_size
        patch_size = self.config.patch_size
        vis = image.copy().astype(np.float32)

        for color, target_mask in zip(_distinct_colors(len(target_masks)), target_masks):
            for idx in target_mask:
                self._blend_patch(vis, idx, grid, patch_size, color)

        for idx in context_mask:
            self._blend_patch(vis, idx, grid, patch_size, (0, 255, 255))

        return np.clip(vis, 0, 255).astype(np.uint8)


# ======================================================================
# 3. Générateur de données Keras
# ======================================================================

class IJEPADataGenerator(_BaseSequence):
    """Générateur de données Keras pour l'entraînement I-JEPA.

    Produit, pour chaque batch, `((contexte_masqué, (masques_cibles,
    longueurs)), labels)` -- des labels factices sont renvoyés si aucun
    label n'est fourni (cas auto-supervisé pur).

    Compatible avec `tf.keras.utils.Sequence` (TensorFlow < 2.16) et
    `keras.utils.PyDataset` (Keras 3 / TensorFlow >= 2.16), les deux
    exposant la même interface `__len__` / `__getitem__` / `on_epoch_end`.
    """

    def __init__(self, images: np.ndarray, labels: Optional[np.ndarray] = None,
                 batch_size: int = 32, img_size: int = 64, patch_size: int = 8,
                 shuffle: bool = True, normalize: bool = True,
                 masking_config: Optional[JEPAMaskingConfig] = None,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)

        if images.ndim != 4:
            raise ValueError(
                f"`images` doit avoir 4 dimensions (N, H, W, C), reçu "
                f"{images.shape}."
            )
        if images.shape[1] != img_size or images.shape[2] != img_size:
            raise ValueError(
                f"`images` a une résolution {images.shape[1:3]}, attendu "
                f"({img_size}, {img_size})."
            )
        if labels is not None and len(labels) != len(images):
            raise ValueError("`labels` et `images` doivent avoir la même longueur.")
        if batch_size < 1:
            raise ValueError("batch_size doit être >= 1.")

        # dtype d'origine conservé (pas de duplication float32 en mémoire) ;
        # la normalisation n'est appliquée que par batch, dans _build_batch.
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.normalize = normalize

        self.config = masking_config or JEPAMaskingConfig.standard(
            img_size=img_size, patch_size=patch_size
        )
        # RNG unique et seedable, utilisé à la fois pour le mélange des
        # indices et pour la génération des masques -- corrige le mélange
        # `random`/`numpy.random` global de la version initiale, qui
        # empêchait toute reproductibilité complète via un seul `seed`.
        self.rng = np.random.default_rng(seed)
        self.mask_generator = JEPAMaskGenerator(self.config, rng=self.rng)

        self.indices = np.arange(len(images))
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.images) / self.batch_size))

    def _apply_mask_to_image(self, image: np.ndarray,
                              context_mask: Sequence[int]) -> np.ndarray:
        grid = self.config.grid_size
        patch_size = self.config.patch_size
        masked = np.zeros_like(image, dtype=np.float32)
        for idx in context_mask:
            row, col = divmod(idx, grid)
            y1, x1 = row * patch_size, col * patch_size
            y2, x2 = y1 + patch_size, x1 + patch_size
            masked[y1:y2, x1:x2] = image[y1:y2, x1:x2]
        return masked

    def _pad_target_masks(self, target_masks_list: List[List[List[int]]]
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """Convertit des listes de longueurs variables en tenseurs
        rectangulaires, avec un tableau de longueurs pour ignorer le
        padding en aval (dans la fonction de perte, par exemple)."""
        batch_size = len(target_masks_list)
        if batch_size == 0:
            return np.zeros((0, 0, 0), dtype=np.int32), np.zeros((0, 0), dtype=np.int32)

        num_targets = len(target_masks_list[0])
        max_patches = max(
            (len(m) for masks in target_masks_list for m in masks), default=0
        )
        max_patches = max(max_patches, 1)  # jamais 0 : évite un tenseur dégénéré

        padded = np.zeros((batch_size, num_targets, max_patches), dtype=np.int32)
        lengths = np.zeros((batch_size, num_targets), dtype=np.int32)
        for i, masks in enumerate(target_masks_list):
            for j, m in enumerate(masks):
                lengths[i, j] = len(m)
                if m:
                    padded[i, j, : len(m)] = m
        return padded, lengths

    def _build_batch(self, idx: int):
        """Construit un batch et renvoie EN PLUS les masques bruts utilisés.

        C'est le correctif central du bug de visualisation : `__getitem__`
        et `visualize_batch` appellent tous les deux cette méthode, et
        partagent donc exactement les mêmes masques -- au lieu que
        `visualize_batch` en régénère indépendamment, comme dans la version
        initiale.
        """
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = self.images[batch_indices].astype(np.float32)
        if self.normalize:
            batch_images = batch_images / 255.0

        context_masks, target_masks_list = self.mask_generator.generate_masks(
            len(batch_images)
        )

        context_images = np.stack([
            self._apply_mask_to_image(img, mask)
            for img, mask in zip(batch_images, context_masks)
        ])

        padded_target_masks, mask_lengths = self._pad_target_masks(target_masks_list)
        target_data = (
            tf.constant(padded_target_masks, dtype=tf.int32),
            tf.constant(mask_lengths, dtype=tf.int32),
        )

        if self.labels is not None:
            batch_labels = tf.constant(self.labels[batch_indices], dtype=tf.int32)
        else:
            batch_labels = tf.zeros(len(batch_indices), dtype=tf.int32)

        inputs = (tf.constant(context_images, dtype=tf.float32), target_data)
        return inputs, batch_labels, batch_indices, context_masks, target_masks_list

    def __getitem__(self, idx: int):
        inputs, labels, _, _, _ = self._build_batch(idx)
        return inputs, labels

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def visualize_batch(self, batch_idx: int = 0, n_images: int = 2,
                         save_path: Optional[str] = None) -> None:
        """Affiche image originale / contexte masqué / cibles / superposition
        pour inspection visuelle, en réutilisant exactement les masques du
        batch réellement produit par `__getitem__` (voir `_build_batch`)."""
        inputs, _, batch_indices, context_masks, target_masks_list = self._build_batch(batch_idx)
        context_images = inputs[0].numpy()
        original_images = self.images[batch_indices].astype(np.uint8)

        n_images = min(n_images, len(context_images))
        fig, axes = plt.subplots(n_images, 4, figsize=(10, 2.5 * n_images))
        if n_images == 1:
            axes = axes.reshape(1, -1)

        display_scale = 255.0 if self.normalize else 1.0

        for i in range(n_images):
            axes[i, 0].imshow(original_images[i])
            axes[i, 0].set_title(f"Originale {i + 1}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(
                np.clip(context_images[i] * display_scale, 0, 255).astype(np.uint8)
            )
            axes[i, 1].set_title("Contexte masqué")
            axes[i, 1].axis("off")

            vis_targets = self.mask_generator.visualize_masks(
                original_images[i], [], target_masks_list[i]
            )
            axes[i, 2].imshow(vis_targets)
            axes[i, 2].set_title(f"Cibles ({self.config.num_targets} blocs)")
            axes[i, 2].axis("off")

            vis_full = self.mask_generator.visualize_masks(
                original_images[i], context_masks[i], target_masks_list[i]
            )
            axes[i, 3].imshow(vis_full)
            axes[i, 3].set_title("Contexte + cibles")
            axes[i, 3].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Figure sauvegardée : %s", save_path)
        plt.show()
        plt.close(fig)