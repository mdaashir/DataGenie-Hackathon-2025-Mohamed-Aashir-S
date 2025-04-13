from Backend.Models.TimeSeries import Arima, Sarimax, Ets, Stl_Ets, Prophet, Lstm, Garch, Xgboost, Naive, Seasonal_Naive

def forecast(data, date_from, date_to, frequency, period, model_name):
    try:
        model_name = model_name.lower()

        if model_name == "arima":
            return Arima.predict(data, date_from, date_to, frequency, period)

        elif model_name == "sarimax":
            return Sarimax.predict(data, date_from, date_to, frequency, period)

        elif model_name == "ets":
            return Ets.predict(data, date_from, date_to, frequency, period)

        elif model_name == "stl+ets":
            return Stl_Ets.predict(data, date_from, date_to, frequency, period)

        elif model_name == "prophet":
            return Prophet.predict(data, date_from, date_to, frequency, period)

        elif model_name == "lstm":
            return Lstm.predict(data, date_from, date_to, frequency, period)

        elif model_name == "garch":
            return Garch.predict(data, date_from, date_to, frequency, period)

        elif model_name == "xgboost":
            return Xgboost.predict(data, date_from, date_to, frequency, period)

        elif model_name == "naive":
            return Naive.predict(data, date_from, date_to, frequency, period)

        elif model_name == "seasonalnaive":
            return Seasonal_Naive.predict(data, date_from, date_to, frequency, period)

        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

    except Exception as e:
        return str(e)