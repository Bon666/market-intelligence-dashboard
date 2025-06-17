# 📊 Market Intelligence Dashboard

This project performs an advanced data analysis of global financial assets using Python. It includes data collection, preprocessing, visualization, risk-return metrics, and statistical modeling such as PCA and KMeans clustering.

## ✅ Features

- Downloads multi-asset financial data from Yahoo Finance
- Computes annual returns, volatility, Sharpe ratios
- Correlation heatmap and pairwise plots
- Principal Component Analysis (PCA) for asset factors
- Clustering of assets using KMeans
- Visual outputs saved in the `graphs/` folder

## 📂 Structure

- `data/`: Contains the downloaded financial data
- `graphs/`: Plots of risk, return, correlation, PCA, clustering
- `notebooks/market_analysis.ipynb`: Full Jupyter notebook with analysis
- `requirements.txt`: Required packages

## ▶️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/market_analysis.ipynb
```

## 📈 Assets Used

- AAPL, MSFT, GOOGL, TSLA, AMZN, META, GLD, BTC-USD, EURUSD=X, ^GSPC

---
