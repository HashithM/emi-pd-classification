# EMI-PD Classification

9-class EMI / partial discharge classification using 1D CNN-LSTM + semi-supervised anomaly detection with convolutional autoencoder (PyTorch)

## Overview

This project implements a complete deep learning pipeline for analyzing electromagnetic interference (EMI) and partial discharge (PD) signals in high-voltage equipment.

Key components:
- **Supervised classification**: 1D CNN + Bidirectional LSTM to classify signals into 9 discharge/fault types
- **Anomaly detection**: Convolutional autoencoder trained only on normal data to detect rare/unseen events via reconstruction error
- **Dataset**: ~11,150 samples, each a normalized 4000-timestep 1D signal
- **Framework**: PyTorch

## Results (update with your numbers)

- **Classification**  
  Validation accuracy: XX%  
  Macro-F1 score: XX  
  Best model checkpoint: `results/best_model.pth`

- **Anomaly Detection**  
  AUC on rare class (treated as anomaly): XX.XX  
  Precision/Recall/F1 at chosen threshold: XX / XX / XX

![Class Distribution](results/emi_class_distribution.png)
![Training & Validation Loss](results/loss_curve.png)
![Validation Accuracy](results/accuracy_curve.png)
<!-- Add more: confusion matrix, reconstruction error histogram, example predictions -->


