import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from Backend.Utils.Data_Processor import parse_date, calculate_forecast_period, get_actual_values, create_result_json

def predict(data, date_from, date_to, frequency, period):
    date_from_ts = parse_date(date_from)
    date_to_ts = parse_date(date_to)
    training_data = data[:date_from_ts]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(training_data[:-1].values.reshape(-1, 1))

    x, y = [], []
    for i in range(10, len(scaled)):
        x.append(scaled[i-10:i])
        y.append(scaled[i])
    x, y = np.array(x), np.array(y)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=30, batch_size=16, verbose=0)

    input_seq = scaled[-10:].reshape(1, 10, 1)
    forecast_period = calculate_forecast_period(date_from_ts, date_to_ts, frequency, period)
    preds = []

    for _ in range(forecast_period):
        pred = model.predict(input_seq, verbose=0)[0, 0]
        preds.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    forecast_index = pd.date_range(start=date_from_ts, periods=forecast_period, freq=frequency)
    forecast = pd.Series(forecast, index=forecast_index)

    actual_values = get_actual_values(data, date_from_ts, frequency, forecast_period)
    result_json = create_result_json(forecast, actual_values)
    mape = mean_absolute_percentage_error(actual_values, forecast)
    return result_json, mape
