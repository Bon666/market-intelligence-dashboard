# Global Market Intelligence Dashboard (Advanced)

An advanced multi-asset market analytics project that produces an interactive HTML dashboard.

## What it does
- Fetches real market data (Yahoo Finance)
- Computes risk–return metrics (annualized return, volatility, Sharpe ratio)
- Builds correlation heatmap and normalized performance comparison
- Applies PCA to learn latent market factors
- Uses KMeans clustering to group assets by behavior
- Generates a single interactive dashboard HTML

## Tech Stack
Python, pandas, NumPy, yfinance, scikit-learn (PCA/KMeans), Plotly

## Run
```bash
pip install -r requirements.txt
python main.py --years 5
