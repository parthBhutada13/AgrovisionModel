# Vision Transformer-Based Multi-Task Crop Mapping and Field Boundary Detection in Heterogeneous Agricultural Landscapes

**Course:** Computer Vision

**Degree:** Bachelor of Technology in Artificial Intelligence & Machine Learning (AY 2025-26)

**Submitted By:**
- Alabhya Sharma (23070126010)
- Parth Bhutada (23070126084)
- Lucy Uguwaneke (23070126169)
- Swapnil Samrat (24070126510)

**Under the Guidance of:** Dr. Sumanto Dutta, Assistant Professor

**Institution:** Department of Artificial Intelligence & Machine Learning, Symbiosis International (Deemed University), Pune – 412115, Maharashtra, India

---

## Certificate

This is to certify that the Project work entitled “Vision Transformer-Based Multi-Task Crop Mapping and Field Boundary Detection in Heterogeneous Agricultural Landscapes” is carried out by the students listed above under the Computer Vision course during the academic year 2025-2026.

(Guide signature block omitted)

---

## Abstract

**Keywords:** Precision Agriculture, Multispectral Remote Sensing, CNN–Transformer Hybrid Model, Vegetation Index Analysis, Explainable AI (XAI), Crop Classification

The AgroVision project proposes a hybrid deep-learning pipeline that combines convolutional neural networks (CNNs) with Vision Transformer-based self-attention to jointly capture local texture and long-range context for crop classification and field boundary detection. A spectral-aware embedding with Squeeze-and-Excitation (SE) blocks and a Feature Pyramid Network (FPN) enable robust multi-scale feature learning. The method is trained and evaluated on the EuroSAT dataset and demonstrates strong performance (reported accuracy 90.83%, F1-score 0.908) along with explainability via Grad-CAM and explicit boundary detection.

---

## Table of Contents
1. Introduction
2. Literature Review
3. Methodology
4. Data Acquisition and Preprocessing
5. Model Architecture
6. Training Strategy
7. Explainability and Evaluation
8. Results
9. Conclusion and Future Work

---

## 1. Introduction

(See full notebook for detailed narrative.)

The project addresses challenges of fragmented field shapes, spectral variability, and the need for both local and global context in agricultural classification. AgroVision fuses CNN and Transformer components, uses vegetation indices, and adds a boundary-detection head for improved spatial precision.

## 2. Problem Statement & Objectives

- Build a hybrid CNN–Transformer model for crop classification in heterogeneous landscapes.
- Improve multi-scale representation via FPN and reduce class confusion with cross-attention.
- Provide explainability (Grad-CAM) and field boundary detection.

## 3. Methodology

Key components:
- Spectral-aware embedding + Band-Group Attention
- CNN backbone + FPN
- CNN–Transformer fusion block
- Cross-attention refinement
- Boundary detection head (Sobel prior + learnable decoder)
- Training with MixUp, label smoothing, and weighted loss

## 4. Data & Preprocessing

Trained on EuroSAT (RGB / Multispectral variants used), images normalized, resized, and augmented (flips, rotations, color jitter, band dropout).

## 5. Results (Summary)

- Reported overall accuracy: 90.83%
- Macro F1 / weighted F1: ≈ 0.908
- AUC (per-class): ≈ 0.93 (reported)
- Explainability: Grad-CAM highlights crop structures and field boundaries
- Boundary detection: Sobel-primed decoder produces localized edge probability maps

Refer to `main.ipynb` and `eval_4class.py` for detailed figures (confusion matrix, ROC, t-SNE, attention maps, band importance, training curves).

## 6. Conclusion & Future Work

AgroVision demonstrates the benefit of hybrid local+global modelling and explicit boundary detection for precision agriculture. Future directions: multispectral extensions, semi/self-supervised learning, real-time optimization and UAV integration.

---

## Figures & Tables
Please open `main.ipynb` to view embedded figures and full experimental details.

---

## Resume Blurb (for inclusion)
- Developed "AgroVision": a hybrid CNN–Transformer model for multispectral crop classification and field boundary detection. Implemented training and evaluation pipelines, explainability (Grad-CAM), and a Streamlit demo for interactive exploration.
