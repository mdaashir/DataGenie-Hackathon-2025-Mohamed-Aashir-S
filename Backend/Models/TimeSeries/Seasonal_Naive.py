import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from Backend.Utils.Data_Processor import parse_date, calculate_forecast_period, get_actual_values, create_result_json

def predict(data, date_from, date_to, frequency, period, seasonality=12):
    try:
        date_from_ts = parse_date(date_from)
        date_to_ts = parse_date(date_to)

        if data.empty:
            raise ValueError(
                "Input data is empty. Please provide valid time series data."
            )

        training_data = data[:date_from_ts]
        if len(training_data) < seasonality:
            raise ValueError(
                f"Not enough data for seasonal naive forecast. Minimum {seasonality} data points required."
            )

        forecast_period = (
            calculate_forecast_period(date_from_ts, date_to_ts, frequency, period) + 1
        )

        actual_values = get_actual_values(
            data, date_from_ts, frequency, forecast_period
        )

        seasonal_pattern = training_data.iloc[-seasonality:].values
        repeats = (forecast_period // seasonality) + 1
        forecast_values = (seasonal_pattern.tolist() * repeats)[:forecast_period]

        forecast_index = pd.date_range(start=date_from_ts, periods=forecast_period, freq=frequency)
        forecast = pd.Series(forecast_values, index=forecast_index)

        result_json = create_result_json(forecast, actual_values)
        mape = mean_absolute_percentage_error(actual_values, forecast)
        return result_json, mape

    except Exception as e:
        print(f"Error during forecast: {str(e)}")
        return None, None
