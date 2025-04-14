import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from Backend.Utils.Data_Processor import parse_date, calculate_forecast_period, get_actual_values, create_result_json

def predict(data, date_from, date_to, frequency, period):
    try:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index.")

        date_from_ts = parse_date(date_from)
        date_to_ts = parse_date(date_to)

        training_data = data[:date_from_ts][:-1]

        if training_data.empty:
            raise ValueError("Training data is empty. Check the date range and data.")

        df = pd.DataFrame({"ds": training_data.index, "y": training_data.values})
        model = Prophet()
        model.fit(df)

        forecast_period = calculate_forecast_period(date_from_ts, date_to_ts, frequency, period)
        future = model.make_future_dataframe(periods=forecast_period, freq=frequency)

        forecast_df = model.predict(future)

        forecast = forecast_df.set_index("ds")["yhat"][-forecast_period:]

        actual_values = get_actual_values(data, date_from_ts, frequency, forecast_period)

        result_json = create_result_json(forecast, actual_values)
        mape = mean_absolute_percentage_error(actual_values, forecast)
        return result_json, mape

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None
