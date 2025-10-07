# FFIR-Net: A Feature Fusion and Inter-stock Relationship Network for Stock Price Movement Prediction
This repository contains the official implementation of the paper "FFIR-Net: A Feature Fusion and Inter-stock Relationship Network for Stock Price Movement Prediction".
## Overview
FFIR-Net is a novel deep learning framework designed for accurate stock price movement prediction by integrating feature fusion and inter-stock relationship learning. The model addresses key challenges in financial market prediction, such as capturing complex temporal patterns, fusing multi-modal data (trading data and investor sentiment), and modeling dynamic inter-stock relationships.

## Key Innovations
FFIR-Net addresses the challenges of stock price movement prediction through three key modules:
1. **Feature Fusion Module**: Integrates historical on-exchange trading data (VOHLC: Volume, Open, High, Low, Close) and investor sentiment data using tensor fusion, followed by a multiscale feature mixer to capture temporal patterns across different time horizons .
2. **Multiscale Feature Mixer (MFM)**:Captures temporal evolution patterns across different time horizons, from short-term fluctuations to long-term trends.
3. **Inter-Stock Relationship Learning (ISRL) Module**: Utilizes a dual-channel hypergraph attention network with group enhancement (HGAT-GE) to model both industry-based static relationships and market-driven dynamic


## Architecture
The FFIR-Net framework consists of three main components:
1.**Feature Fusion Module** (Tensor Fusion + Multiscale Feature Mixer)
2.**Inter-Stock Relationship Learning Module** (Dual-channel HGAT-GE for static and dynamic relationships)
3.**Output Mapping Module** (Binary classification for price movement prediction)
## Datasets
The model is evaluated on two real-world datasets:
- **CSI100E**: Contains 516 trading days of data (Nov 2017 - Dec 2019) with VOHLC trading data and sentiment features (positive/negative proportions, sentiment intensity) from Chinese financial news .
- **S&P500**: Includes 700 trading days of data (Feb 2011 - Nov 2013) with trading records and 7-dimensional sentiment features from Reuters/Bloomberg news .

Industry relationships for both datasets are sourced from the iFinD financial data terminal.

## Results
FFIR-Net outperforms state-of-the-art baselines (e.g., LSTM, GCN, AD-GAT, FinHGNN,VGC-GAN,MagicNet) in both classification metrics (accuracy, AUC, F1-score) and practical trading profitability (IRR, Sharpe Ratio) across bull, bear, and sideways market conditions.

Due to the needs of subsequent work, we are currently providing a simplified version of the model and example implementations. The full project code will be gradually released in future updates.

We appreciate your understanding. If you have any specific questions or requests regarding the code, please feel free to contact us.
