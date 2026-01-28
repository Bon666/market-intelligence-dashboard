
---

## `main.py`（高级完整代码，直接可跑）
```python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -----------------------------
# Utilities
# -----------------------------
def ensure_dirs():
    out_dir = Path("outputs")
    data_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    return out_dir, data_dir


def fetch_prices(tickers, years, data_dir):
    tickers = [t.upper() for t in tickers]
    cache = data_dir / f"prices_{'-'.join(tickers)}_{years}y.csv"

    if cache.exists():
        return pd.read_csv(cache, parse_dates=["Date"]).set_index("Date")

    df = yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False)["Close"]
    df.index.name = "Date"
    df.to_csv(cache)
    return df


def annualized_metrics(returns: pd.DataFrame, rf: float):
    """
    returns: daily returns
    rf: annual risk-free rate (e.g., 0.02)
    """
    mu = returns.mean() * 252
    vol = returns.std() * np.sqrt(252)
    sharpe = (mu - rf) / vol.replace(0, np.nan)
    out = pd.DataFrame({"ann_return": mu, "ann_vol": vol, "sharpe": sharpe})
    return out.dropna()


def normalize_prices(prices: pd.DataFrame):
    return prices / prices.iloc[0]


def compute_pca_and_clusters(returns: pd.DataFrame, n_components: int, n_clusters: int, seed: int = 42):
    """
    PCA on standardized returns. KMeans on PCA space.
    """
    # Drop assets with too many missing values
    returns = returns.dropna(axis=1, how="any")

    X = returns.values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)

    # Loadings (how each asset contributes to components) - using PCA components on features
    # For time-series returns: features are assets if we transpose. Better:
    # PCA on assets behavior => PCA on correlation matrix style: use returns.T (assets x time)
    X_assets = returns.T.values  # assets x time
    X_assets_scaled = StandardScaler().fit_transform(X_assets)

    pca_assets = PCA(n_components=n_components, random_state=seed)
    asset_pcs = pca_assets.fit_transform(X_assets_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    clusters = kmeans.fit_predict(asset_pcs)

    pca_summary = {
        "explained_variance_ratio": pca_assets.explained_variance_ratio_,
        "asset_pcs": pd.DataFrame(asset_pcs, index=returns.columns, columns=[f"PC{i+1}" for i in range(n_components)]),
        "clusters": pd.Series(clusters, index=returns.columns, name="cluster"),
        "loadings": pd.DataFrame(pca_assets.components_.T,
                                index=[f"t{i}" for i in range(returns.shape[0])],
                                columns=[f"PC{i+1}" for i in range(n_components)])
    }

    return pca_summary


# -----------------------------
# Dashboard build
# -----------------------------
def build_dashboard(prices, returns, metrics, corr, asset_pcs, clusters, out_dir):
    # 1) Normalized performance
    norm = normalize_prices(prices)

    perf_fig = px.line(
        norm,
        title="Normalized Performance (Start = 1.0)",
        labels={"value": "Normalized Price", "Date": "Date"},
    )

    # 2) Correlation heatmap
    heat_fig = px.imshow(
        corr,
        text_auto=False,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        title="Asset Return Correlation Heatmap"
    )

    # 3) Risk-return scatter (size=Sharpe, color=cluster)
    rr = metrics.copy()
    rr["cluster"] = clusters.reindex(rr.index)
    rr_fig = px.scatter(
        rr.reset_index().rename(columns={"index": "asset"}),
        x="ann_vol",
        y="ann_return",
        color="cluster",
        size="sharpe",
        hover_name="asset",
        title="Risk–Return Map (Color=Cluster, Size=Sharpe)"
    )

    # 4) PCA scatter (PC1 vs PC2)
    pcs_df = asset_pcs.copy()
    pcs_df["cluster"] = clusters
    pcs_df["asset"] = pcs_df.index

    pca_fig = px.scatter(
        pcs_df,
        x="PC1",
        y="PC2",
        color="cluster",
        hover_name="asset",
        title="PCA Asset Map (PC1 vs PC2)"
    )

    # Combine into one HTML via subplots-like layout (simple approach):
    # We'll create a single page with multiple figures using Plotly's HTML concatenation.
    html_parts = []
    html_parts.append("<h1>Global Market Intelligence Dashboard</h1>")
    html_parts.append("<p>Multi-asset analytics: performance, correlation, risk-return, PCA & clustering.</p>")

    for fig in [perf_fig, heat_fig, rr_fig, pca_fig]:
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    html = "\n".join(html_parts)
    out_path = out_dir / "dashboard.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Global Market Intelligence Dashboard (Advanced)")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["SPY", "QQQ", "DIA", "IWM", "GLD", "SLV", "USO", "TLT", "EFA", "EEM"],
        help="List of tickers (ETFs/indices/stocks)."
    )
    parser.add_argument("--rf", type=float, default=0.02, help="Annual risk-free rate")
    parser.add_argument("--pca_components", type=int, default=2)
    parser.add_argument("--clusters", type=int, default=3)
    args = parser.parse_args()

    out_dir, data_dir = ensure_dirs()

    prices = fetch_prices(args.tickers, args.years, data_dir).dropna()
    if prices.empty:
        raise RuntimeError("No price data returned. Check tickers or internet connection.")

    returns = prices.pct_change().dropna()

    metrics = annualized_metrics(returns, rf=args.rf)
    corr = returns.corr()

    pca_summary = compute_pca_and_clusters(
        returns=returns,
        n_components=args.pca_components,
        n_clusters=args.clusters
    )

    asset_pcs = pca_summary["asset_pcs"]
    clusters = pca_summary["clusters"]

    # Save CSVs
    metrics.to_csv(out_dir / "risk_return_metrics.csv")
    asset_pcs.to_csv(out_dir / "pca_scores.csv")
    clusters.to_csv(out_dir / "cluster_assignments.csv")

    # Build interactive dashboard
    build_dashboard(
        prices=prices[metrics.index],      # align
        returns=returns[metrics.index],
        metrics=metrics,
        corr=corr.loc[metrics.index, metrics.index],
        asset_pcs=asset_pcs,
        clusters=clusters,
        out_dir=out_dir
    )


if __name__ == "__main__":
    main()
