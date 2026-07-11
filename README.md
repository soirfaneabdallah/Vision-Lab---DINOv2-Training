# 🧠 Vision Lab – Entraînement DINOv2

<p align="center">
  <img src="assets/dinov2_logo.png" alt="DINOv2 Training" width="200"/>
</p>

<p align="center">
  <strong>Pipeline d'entraînement du classifieur DINOv2 pour l'application Vision Lab</strong>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.9%2B-yellow" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/framework-PyTorch-red" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/status-production-brightgreen" alt="Status"></a>
</p>

---

## 📖 À propos

Ce dépôt contient le code d'entraînement du modèle de classification utilisé dans **Vision Lab**. Il permet de fine-tuner un modèle **DINOv2** (Vision Transformer pré-entraîné par Meta AI) sur un jeu de données personnalisé de poissons de récif et de déchets marins.

Le modèle entraîné est ensuite exporté au format **ONNX** pour une intégration optimale dans l'application de bureau Vision Lab, où il est utilisé en production avec **ONNX Runtime**.

> **Pourquoi DINOv2 ?**  
> Les modèles DINOv2 produisent des caractéristiques visuelles haute performance qui peuvent être directement utilisées avec des classifieurs aussi simples que des couches linéaires. Ces caractéristiques sont robustes et performantes dans tous les domaines sans nécessiter de fine-tuning lourd.

---

## ✨ Fonctionnalités principales

| Fonctionnalité | Description |
| :--- | :--- |
| **🧠 Fine-tuning DINOv2** | Entraînement supervisé du classifieur sur un dataset personnalisé |
| **🔒 Backbone gelé ou non** | Possibilité de geler le backbone DINOv2 (linear probing) ou de le fine-tuner complètement |
| **📊 Data augmentation** | Augmentation des données pour améliorer la généralisation |
| **⚖️ Gestion des classes déséquilibrées** | Class Balanced Loss / échantillonnage des classes rares |
| **📈 Suivi des expériences** | Logging des métriques (accuracy, loss, F1-score) via TensorBoard |
| **🔁 Early stopping** | Arrêt automatique pour éviter le sur-apprentissage |
| **📦 Export ONNX** | Conversion du modèle entraîné au format ONNX pour l'inférence en production |
| **🧪 Validation automatique** | Split entraînement / validation avec évaluation périodique |

---

## 🗂️ Structure du dataset

Le script attend un dataset organisé de la manière suivante :
