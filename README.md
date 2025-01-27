# Hydrological Contrastive Learning

A self-supervised contrastive learning framework for hydrological time-series data. This project uses an LSTM-based encoder to learn meaningful representations of hydrographs (river discharge over time), which can be utilized for tasks like catchment classification.

## Introduction

Understanding hydrological patterns is crucial for water resource management, flood prediction, and environmental conservation. This project implements a self-supervised contrastive learning approach to analyze hydrological time-series data. By learning embeddings of hydrographs without explicit labels, the model can capture intrinsic patterns and similarities in the data.

## Features

- **Contrastive Learning Framework**: Utilizes self-supervised learning to generate meaningful embeddings.
- **LSTM-based Encoder**: Captures temporal dependencies in hydrological data.
- **Flexible Data Loader**: Custom dataset and data loader for efficient sampling.

## Requirements

- Python 3.7 or higher
- PyTorch 1.7 or higher
- NumPy
- Pandas
- scikit-learn
- LightGBM
- info-nce-pytorch (or your custom implementation of InfoNCE loss)
- Other dependencies will appear here soon...
 
