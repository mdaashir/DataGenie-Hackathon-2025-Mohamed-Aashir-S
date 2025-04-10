import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def generate_series(model, length=200, seed=42):
    np.random.seed(seed)
    t = np.arange(length)

    trend = np.zeros(length)
    seasonality = np.zeros(length)
    noise = np.random.normal(0, 1, length)

    if model == "ARIMA":
        series = np.random.randn(length).cumsum()

    elif model == "SARIMAX":
        trend = 0.05 * t
        seasonality = 5 * np.sin(2 * np.pi * t / 12)
        ar_component = (
            pd.Series(np.random.randn(length)).rolling(3, min_periods=1).mean()
        )
        series = trend + seasonality + ar_component + noise

    elif model == "ETS":
        trend = np.linspace(10, 20, length)
        seasonality = 4 * np.sin(2 * np.pi * t / 6)
        series = trend + seasonality + noise * 0.2

    elif model == "STL+ETS":
        trend = np.linspace(0, 10, length)
        trend[length // 2 :] += 5  # Structural break
        seasonality = 6 * np.sin(2 * np.pi * t / 12)
        series = trend + seasonality + np.random.normal(0, 0.5, length)

    elif model == "Prophet":
        trend = np.piecewise(
            t,
            [t < length // 2, t >= length // 2],
            [lambda x: x * 0.1, lambda x: x * 0.3],
        )
        seasonality = 5 * np.sin(2 * np.pi * t / 7)
        missing = np.random.choice(length, size=5, replace=False)
        series = trend + seasonality + noise
        series[missing] = np.nan
        series = pd.Series(series).interpolate().fillna(method="bfill")

    elif model == "LSTM":
        base = np.sin(t / 3) + np.log1p(t)
        noise = np.random.normal(0, np.linspace(0.1, 1.5, length))
        series = base + noise

    elif model == "GARCH":
        volatility = np.linspace(0.2, 2.0, length)
        returns = np.random.normal(0, volatility)
        series = returns.cumsum()

    elif model == "XGBoost":
        trend = 0.03 * (t**2)
        rolling = (
            pd.Series(np.random.randn(length)).rolling(window=5, min_periods=1).mean()
        )
        series = trend + rolling + noise

    elif model == "Naive":
        series = np.random.randn(length).cumsum()

    elif model == "SeasonalNaive":
        seasonality = 6 * np.sin(2 * np.pi * t / 12)
        series = seasonality + np.random.normal(0, 0.2, length)

    else:
        raise ValueError("Unknown model")

    return pd.DataFrame({"timestamp": t, "value": series})


def generate_dataset(per_model=500, length=200, output_dir="synthetic_data"):
    models = [
        "ARIMA",
        "SARIMAX",
        "ETS",
        "STL+ETS",
        "Prophet",
        "LSTM",
        "GARCH",
        "XGBoost",
        "Naive",
        "SeasonalNaive",
    ]

    os.makedirs(output_dir, exist_ok=True)

    for model in tqdm(models, desc="Generating time series"):
        for i in range(per_model):
            df = generate_series(model, length=length, seed=i)
            filename = f"{model}_series_{i}.csv"
            df.to_csv(os.path.join(output_dir, filename), index=False)

    print(f"\nGenerated {per_model * len(models)} series in '{output_dir}/'")


if __name__ == "__main__":
    generate_dataset()
