import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error
from Backend.Utils.Data_Processor import parse_date, calculate_forecast_period, get_actual_values, create_result_json

def predict(data, date_from, date_to, frequency, period, seasonal_periods=12):
    try:
        date_from_ts = parse_date(date_from)
        date_to_ts = parse_date(date_to)

        if data.empty:
            raise ValueError(
                "Input data is empty. Please provide valid time series data."
            )
        if len(data) < seasonal_periods:
            raise ValueError(
                f"Not enough data points. At least {seasonal_periods} data points are required for seasonal forecasting."
            )

        training_data = data[:date_from_ts]

        model = ExponentialSmoothing(training_data[:-1], trend="add", seasonal="add", seasonal_periods=seasonal_periods)
        model_fit = model.fit()

        forecast_period = calculate_forecast_period(date_from_ts, date_to_ts, frequency, period)
        actual_values = get_actual_values(data, date_from_ts, frequency, forecast_period)

        forecast = model_fit.forecast(steps=forecast_period)
        forecast_index = pd.date_range(start=date_from_ts, periods=forecast_period, freq=frequency)
        forecast_series = pd.Series(forecast.values, index=forecast_index)

        result_json = create_result_json(forecast_series, actual_values)
        mape = mean_absolute_percentage_error(actual_values, forecast_series)
        return result_json, mape

    except Exception as e:
        print(f"Error during forecasting: {e}")
        return None, None
