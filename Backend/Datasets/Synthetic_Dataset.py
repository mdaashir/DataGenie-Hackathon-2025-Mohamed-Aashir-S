import os
import random
import numpy as np
import logging
from tqdm import tqdm
import pandas as pd
from datetime import datetime, timedelta

from darts.utils.timeseries_generation import (
    linear_timeseries,
    sine_timeseries,
    random_walk_timeseries,
    gaussian_timeseries,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(output_dir, log_name="generation.log"):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w")],
    )
    logging.info("Logging initialized.")
    return log_path


def format_and_save(df, current_date, length, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    timestamps = []
    base_datetime = datetime.combine(current_date.date(), datetime.min.time())
    for i in range(length):
        base_dt = base_datetime + timedelta(days=i)
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        full_dt = base_dt + timedelta(hours=hour, minutes=minute, seconds=second)
        timestamps.append(full_dt.strftime("%Y-%m-%dT%H:%M:%S"))
    df["timestamp"] = timestamps
    df.columns = ["point_values", "timestamp"]
    df = df[["timestamp", "point_values"]]
    df.to_csv(file_path, index=False)


def generate_dataset(
    per_model=500,
    length=100,
    output_dir="synthetic_data",
    log_dir="../logs",
    seed=42,
    start_date="2020-01-01",
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = setup_logging(log_dir)

    total_series = per_model * length
    current_date = datetime.strptime(start_date, "%Y-%m-%d")

    logging.info(f"Starting generation of {total_series} synthetic time series...")

    try:
        for i in tqdm(range(per_model), desc="Generating synthetic series per model"):
            logging.info(f"Generating set {i + 1}/{per_model}")

            try:
                # ARIMA
                arima = random_walk_timeseries(length=length) + linear_timeseries(
                    length=length
                )
                format_and_save(
                    arima.to_series().to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/ARIMA/ARIMA_series_{i}.csv",
                )

                # SARIMAX
                sarimax = sine_timeseries(length=length) + linear_timeseries(
                    length=length
                )
                format_and_save(
                    sarimax.to_series().to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/SARIMAX/SARIMAX_series_{i}.csv",
                )

                # ETS
                ets = linear_timeseries(
                    length=length, start_value=10, end_value=30
                ) + sine_timeseries(length=length, value_amplitude=2.5)
                format_and_save(
                    ets.to_series().to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/ETS/ETS_series_{i}.csv",
                )

                # STL+ETS
                stl = linear_timeseries(
                    length=length, start_value=0, end_value=20
                ) + sine_timeseries(length=length, value_amplitude=3)
                stl_vals = stl.to_series()
                stl_vals.iloc[length // 2 :] += 5
                format_and_save(
                    stl_vals.to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/STL+ETS/STL+ETS_series_{i}.csv",
                )

                # Prophet
                part1 = linear_timeseries(
                    length=length // 2, start_value=0, end_value=10
                )
                next_start = part1.end_time() + part1.freq
                part2 = linear_timeseries(
                    length=length // 2, start_value=10, end_value=40, start=next_start
                )
                prophet = part1.append(part2) + sine_timeseries(length=length)
                format_and_save(
                    prophet.to_series().to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/Prophet/Prophet_series_{i}.csv",
                )

                # LSTM
                lstm = sine_timeseries(
                    length=length, value_amplitude=5
                ) + gaussian_timeseries(length=length, std=1.0)
                format_and_save(
                    lstm.to_series().to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/LSTM/LSTM_series_{i}.csv",
                )

                # GARCH
                garch = gaussian_timeseries(length=length, std=1.0)
                garch_vals = garch.to_series()
                garch_vals.iloc[length // 2 :] += pd.Series(
                    [j * 0.1 for j in range(length // 2)],
                    index=garch_vals.index[length // 2 :],
                )
                format_and_save(
                    garch_vals.to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/GARCH/GARCH_series_{i}.csv",
                )

                # XGBoost
                xgb = sine_timeseries(
                    length=length, value_frequency=0.08
                ) + gaussian_timeseries(length=length, std=0.5)
                format_and_save(
                    xgb.to_series().to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/XGBoost/XGBoost_series_{i}.csv",
                )

                # Naive
                naive = random_walk_timeseries(length=length)
                format_and_save(
                    naive.to_series().to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/Naive/Naive_series_{i}.csv",
                )

                # Seasonal Naive
                seasonal = sine_timeseries(
                    length=length, value_amplitude=6.0, value_frequency=0.1
                )
                format_and_save(
                    seasonal.to_series().to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/SeasonalNaive/SeasonalNaive_series_{i}.csv",
                )

            except Exception as e:
                logging.error(
                    f"Error generating series for index {i}: {e}", exc_info=True
                )

            current_date = current_date + timedelta(days=length)

    except Exception as e:
        logging.critical(
            "Fatal error during dataset generation! " + str(e), exc_info=True
        )

    logging.info(
        f"\nDone! {total_series} time series saved in '{output_dir}/'. Log: '{log_file_path}'"
    )


if __name__ == "__main__":
    generate_dataset(start_date="0001-01-01", per_model=5000, length=730)
