import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy, normaltest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import welch
import pywt
import nolds

def extract_features(x):
    # Rename and standardize
    x = x.copy()
    x.columns = ["timestamp", "point_values"]
    x["point_values"] = x["point_values"].astype(float)

    # Scale
    scaler = StandardScaler()
    values = scaler.fit_transform(x["point_values"].values.reshape(-1, 1)).flatten()
    x["scaled"] = values

    # Store base series
    y = x["scaled"]
    t = np.arange(len(y)).reshape(-1, 1)

    # Basic stats
    mean = np.mean(y)
    std = np.std(y)
    var = np.var(y)
    skewness = skew(y)
    kurt = kurtosis(y)
    mode = pd.Series(y).mode().iloc[0]
    median = np.median(y)

    # Stationarity
    adf_stat, adf_pval, _, _, _, _ = adfuller(y)
    stationary = adf_pval <= 0.05

    # Trend
    linreg = LinearRegression().fit(t, y)
    slope = linreg.coef_[0]
    poly = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(t, y)
    poly_coef = poly.named_steps["linearregression"].coef_.tolist()

    # Differencing
    diff = np.diff(y)
    diff_stats = {
        "diff_mean": np.mean(diff),
        "diff_std": np.std(diff),
        "diff_skew": skew(diff),
        "diff_kurtosis": kurtosis(diff),
        "diff_entropy": entropy(np.histogram(diff, bins=10)[0] + 1),
    }

    # Rolling window volatility
    roll_std = pd.Series(y).rolling(window=5).std().dropna()
    vol_mean = roll_std.mean()
    vol_trend = roll_std.iloc[-1] - roll_std.iloc[0]

    # ACF/PACF
    acf_vals = acf(y, nlags=5)
    pacf_vals = pacf(y, nlags=5)
    acf_peaks = int(sum(np.abs(acf_vals[1:]) > 0.25))
    dominant_acf_lag = int(np.argmax(np.abs(acf_vals[1:])) + 1)
    pacf_peaks = int(sum(np.abs(pacf_vals[1:]) > 0.25))

    # Seasonal decomposition
    try:
        result = seasonal_decompose(
            y, period=12, model="additive", extrapolate_trend="freq"
        )
        seasonal = result.seasonal
        trend = result.trend
        resid = result.resid
    except:
        seasonal = trend = resid = pd.Series([0] * len(y))

    seasonal_stats = {
        "seasonal_strength": np.var(seasonal),
        "seasonal_max": np.max(seasonal),
        "seasonal_min": np.min(seasonal),
    }

    trend_stats = {
        "trend_strength": np.var(trend),
        "trend_max": np.nanmax(trend),
        "trend_min": np.nanmin(trend),
    }

    residual_stats = {
        "residual_var": np.var(resid),
        "residual_max": np.nanmax(resid),
        "residual_min": np.nanmin(resid),
    }

    # Spectral features (FFT)
    fft = np.fft.fft(y)
    fft_freq = np.fft.fftfreq(len(y))
    peak_freq = float(fft_freq[np.argmax(np.abs(fft))])

    # PSD / Welch
    f_welch, psd = welch(y, nperseg=min(64, len(y)))
    psd_entropy = entropy(psd + 1e-6)

    # Wavelet
    ca, cd = pywt.dwt(y, "db1")
    wavelet_stats = {
        "cA_mean": np.mean(ca),
        "cD_var": np.var(cd),
    }

    # Complexity / Entropy
    try:
        sampen = nolds.sampen(y)
    except:
        sampen = 0.0

    # Lag correlation
    lag1_corr = pd.Series(y).autocorr(lag=1)
    lag2_corr = pd.Series(y).autocorr(lag=2)

    # Residual normality
    try:
        resid_pval = float(normaltest(resid.dropna())[1])
    except:
        resid_pval = 1.0

    # Compile all features
    features = {
        "mean": mean,
        "std_dev": std,
        "variance": var,
        "skewness": skewness,
        "kurtosis": kurt,
        "mode": mode,
        "median": median,
        "stationary_adf_pval": adf_pval,
        "stationary": stationary,
        "slope": slope,
        "vol_mean": vol_mean,
        "vol_trend": vol_trend,
        "acf_peak_count": acf_peaks,
        "pacf_peak_count": pacf_peaks,
        "dominant_acf_lag": dominant_acf_lag,
        "seasonal_strength": seasonal_stats["seasonal_strength"],
        "trend_strength": trend_stats["trend_strength"],
        "residual_var": residual_stats["residual_var"],
        "peak_frequency_fft": peak_freq,
        "spectral_entropy": psd_entropy,
        "sample_entropy": sampen,
        "lag1_corr": lag1_corr,
        "lag2_corr": lag2_corr,
        "residuals_normality_pval": resid_pval,
        "wavelet_cA_mean": wavelet_stats["cA_mean"],
        "wavelet_cD_var": wavelet_stats["cD_var"],
    }

    # Add flattened poly coefficients and ACF/PACF
    for i, val in enumerate(poly_coef):
        features[f"poly_coef_{i}"] = val
    for i, val in enumerate(acf_vals):
        features[f"acf_lag_{i}"] = val
    for i, val in enumerate(pacf_vals):
        features[f"pacf_lag_{i}"] = val
    features.update(diff_stats)

    return pd.Series(features)
