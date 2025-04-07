# Assignment 3: Machine Learning Pipeline and Hyperparameter Tuning

## Overview

This assignment explores the construction and evaluation of machine learning pipelines using scikit-learn on two popular image datasets: **MNIST** and **Fashion-MNIST**. The tasks include:

- Dataset loading and preprocessing
- Dimensionality reduction with **PCA** and **LDA**
- Classification using **Support Vector Classifier (SVC)** with different kernels
- Hyperparameter tuning with **GridSearchCV**
- Performance evaluation and comparison using confusion matrices and analysis

---

## Datasets

### MNIST
- 70,000 grayscale images (28×28) of handwritten digits (0–9)
- 60,000 training samples, 10,000 test samples
- Loaded from IDX format using `idx2numpy`

### Fashion-MNIST
- 70,000 grayscale images (28×28) of fashion items (10 classes)
- 60,000 training samples, 10,000 test samples
- Loaded similarly from IDX format

---

## Requirements

- Python 3.7+
- NumPy
- scikit-learn
- matplotlib
- idx2numpy

Install dependencies via:

```bash
pip install numpy scikit-learn matplotlib idx2numpy
