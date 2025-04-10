import os
import random
import numpy as np
import logging
from tqdm import tqdm
import pandas as pd
from datetime import timedelta

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
    log_path = os.path.join(output_dir, log_name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w")],
    )
    logging.info("Logging initialized.")
    return log_path


def format_and_save(df, current_date, length, file_path):
    date_index = pd.date_range(start=current_date, periods=length, freq="D")
    current_date += timedelta(days=length)
    df.index = date_index
    df.index.name = "timestamp"
    df.columns = ["point_values"]
    df.reset_index().to_csv(file_path, index=False, date_format="%Y-%m-%dT%H:%M:%S")
    return current_date


def generate_darts_dataset(
    per_model=500,
    length=100,
    output_dir="synthetic_data",
    seed=42,
    start_date="2020-01-01",
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = setup_logging(output_dir)

    total_series = per_model * 10
    current_date = pd.to_datetime(start_date)

    logging.info(f"Starting generation of {total_series} synthetic time series...")

    try:
        for i in tqdm(range(per_model), desc="Generating synthetic series per model"):
            logging.info(f"Generating set {i + 1}/{per_model}")

            try:
                # ARIMA
                arima = random_walk_timeseries(length=length) + linear_timeseries(
                    length=length
                )
                current_date = format_and_save(
                    arima.pd_series().to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/ARIMA_series_{i}.csv",
                )

                # SARIMAX
                sarimax = sine_timeseries(length=length) + linear_timeseries(
                    length=length
                )
                current_date = format_and_save(
                    sarimax.pd_series().to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/SARIMAX_series_{i}.csv",
                )

                # ETS
                ets = linear_timeseries(
                    length=length, start_value=10, end_value=30
                ) + sine_timeseries(length=length, value_amplitude=2.5)
                current_date = format_and_save(
                    ets.pd_series().to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/ETS_series_{i}.csv",
                )

                # STL+ETS
                stl = linear_timeseries(
                    length=length, start_value=0, end_value=20
                ) + sine_timeseries(length=length, value_amplitude=3)
                stl_vals = stl.pd_series()
                stl_vals.iloc[length // 2 :] += 5
                current_date = format_and_save(
                    stl_vals.to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/STL+ETS_series_{i}.csv",
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
                current_date = format_and_save(
                    prophet.pd_series().to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/Prophet_series_{i}.csv",
                )

                # LSTM
                lstm = sine_timeseries(
                    length=length, value_amplitude=5
                ) + gaussian_timeseries(length=length, std=1.0)
                current_date = format_and_save(
                    lstm.pd_series().to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/LSTM_series_{i}.csv",
                )

                # GARCH
                garch = gaussian_timeseries(length=length, std=1.0)
                garch_vals = garch.pd_series()
                garch_vals.iloc[length // 2 :] += pd.Series(
                    [j * 0.1 for j in range(length // 2)],
                    index=garch_vals.index[length // 2 :],
                )
                current_date = format_and_save(
                    garch_vals.to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/GARCH_series_{i}.csv",
                )

                # XGBoost
                xgb = sine_timeseries(
                    length=length, value_frequency=0.08
                ) + gaussian_timeseries(length=length, std=0.5)
                current_date = format_and_save(
                    xgb.pd_series().to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/XGBoost_series_{i}.csv",
                )

                # Naive
                naive = random_walk_timeseries(length=length)
                current_date = format_and_save(
                    naive.pd_series().to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/Naive_series_{i}.csv",
                )

                # Seasonal Naive
                seasonal = sine_timeseries(
                    length=length, value_amplitude=6.0, value_frequency=0.1
                )
                current_date = format_and_save(
                    seasonal.pd_series().to_frame(),
                    current_date,
                    length,
                    f"{output_dir}/SeasonalNaive_series_{i}.csv",
                )

            except Exception as e:
                logging.error(
                    f"Error generating series for index {i}: {e}", exc_info=True
                )

    except Exception as e:
        logging.critical(
            "Fatal error during dataset generation! " + str(e), exc_info=True
        )

    logging.info(
        f"\nDone! {total_series} time series saved in '{output_dir}/'. Log: '{log_file_path}'"
    )


if __name__ == "__main__":
    generate_darts_dataset(start_date="2021-01-01")
