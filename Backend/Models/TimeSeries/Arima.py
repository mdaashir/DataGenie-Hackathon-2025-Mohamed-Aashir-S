import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
from Backend.Utils.Data_Processor import parse_date, calculate_forecast_period, get_actual_values, create_result_json

def predict(data, date_from, date_to, frequency, period):
    date_from_ts = parse_date(date_from)
    date_to_ts = parse_date(date_to)
    training_data = data[:date_from_ts]

    model = ARIMA(training_data.iloc[:-1], order=(2, 2, 2))
    model_fit = model.fit()

    forecast_period = calculate_forecast_period(date_from_ts, date_to_ts, frequency, period)
    actual_values = get_actual_values(data, date_from_ts, frequency, forecast_period)

    forecast = model_fit.forecast(steps=forecast_period)
    forecast_index = pd.date_range(start=date_from_ts, periods=forecast_period, freq=frequency)
    forecast = pd.Series(forecast.values, index=forecast_index)

    result_json = create_result_json(forecast, actual_values)
    mape = mean_absolute_percentage_error(actual_values, forecast)
    return result_json, mape