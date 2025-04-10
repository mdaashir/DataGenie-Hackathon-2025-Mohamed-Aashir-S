import os
import random
import numpy as np
import logging
from tqdm import tqdm
import pandas as pd

from darts.utils.timeseries_generation import (
    linear_timeseries,
    sine_timeseries,
    random_walk_timeseries,
    gaussian_timeseries,
)

def save_timestamp(ts, path, length):
    df = ts.pd_series().to_frame(name="value")
    df.index = pd.date_range(start=pd.Timestamp.today().normalize(), periods=length, freq='D')
    df.to_csv(path)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(output_dir, log_name="generation.log"):
    log_path = os.path.join(output_dir, log_name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
        ],
    )
    logging.info("Logging initialized.")
    return log_path


def generate_darts_dataset(
    per_model=500, length=100, output_dir="synthetic_data", seed=42
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = setup_logging(output_dir)

    total_series = per_model * 10
    logging.info(f"Starting generation of {total_series} synthetic time series...")

    try:
        for i in tqdm(range(per_model), desc="Generating synthetic series per model"):
            logging.info(f"Generating set {i + 1}/{per_model}")

            try:
                # ARIMA
                arima = random_walk_timeseries(length=length) + linear_timeseries(
                    length=length
                )
                save_timestamp(arima, f"{output_dir}/ARIMA_series_{i}.csv", length)
                # arima.to_series().to_csv(
                #     f"{output_dir}/ARIMA_series_{i}.csv", header=False
                # )

                # SARIMAX
                sarimax = sine_timeseries(length=length) + linear_timeseries(
                    length=length
                )
                sarimax.to_series().to_csv(
                    f"{output_dir}/SARIMAX_series_{i}.csv", header=False
                )

                # ETS
                ets = linear_timeseries(
                    length=length, start_value=10, end_value=30
                ) + sine_timeseries(length=length, value_amplitude=2.5)
                ets.to_series().to_csv(f"{output_dir}/ETS_series_{i}.csv", header=False)

                # STL+ETS
                stl = linear_timeseries(
                    length=length, start_value=0, end_value=20
                ) + sine_timeseries(length=length, value_amplitude=3)
                stl_shift = stl.to_series()
                stl_shift.iloc[length // 2 :] += 5
                stl_shift.to_csv(f"{output_dir}/STL+ETS_series_{i}.csv", header=False)

                # Prophet
                part1 = linear_timeseries(
                    length=length // 2, start_value=0, end_value=10
                )
                next_start = part1.end_time() + part1.freq
                part2 = linear_timeseries(
                    length=length // 2, start_value=10, end_value=40, start=next_start
                )
                prophet = part1.append(part2) + sine_timeseries(length=length)
                prophet.to_series().to_csv(
                    f"{output_dir}/Prophet_series_{i}.csv", header=False
                )

                # LSTM
                lstm = sine_timeseries(
                    length=length, value_amplitude=5
                ) + gaussian_timeseries(length=length, std=1.0)
                lstm.to_series().to_csv(
                    f"{output_dir}/LSTM_series_{i}.csv", header=False
                )

                # GARCH
                garch = gaussian_timeseries(length=length, std=1.0)
                garch_vals = garch.to_series()
                garch_vals.iloc[length // 2 :] += pd.Series(
                    [j * 0.1 for j in range(length // 2)],
                    index=garch_vals.index[length // 2 :],
                )
                garch_vals.to_csv(f"{output_dir}/GARCH_series_{i}.csv", header=False)

                # XGBoost
                xgb = sine_timeseries(
                    length=length, value_frequency=0.08
                ) + gaussian_timeseries(length=length, std=0.5)
                xgb.to_series().to_csv(
                    f"{output_dir}/XGBoost_series_{i}.csv", header=False
                )

                # Naive
                naive = random_walk_timeseries(length=length)
                naive.to_series().to_csv(
                    f"{output_dir}/Naive_series_{i}.csv", header=False
                )

                # Seasonal Naive
                seasonal = sine_timeseries(
                    length=length, value_amplitude=6.0, value_frequency=0.1
                )
                seasonal.to_series().to_csv(
                    f"{output_dir}/SeasonalNaive_series_{i}.csv", header=False
                )

            except Exception as e:
                logging.error(
                    f"Error generating series for index {i}: {e}", exc_info=True
                )

    except Exception as e:
        logging.critical(
            "Fatal error during dataset generation!" + str(e), exc_info=True
        )

    logging.info(
        f"\n Done! {total_series} time series saved in '{output_dir}/'. Log: '{log_file_path}'"
    )


if __name__ == "__main__":
    generate_darts_dataset()
