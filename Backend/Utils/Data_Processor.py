import pandas as pd
from datetime import datetime

def parse_date(date_str):
    return pd.Timestamp(datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S"))

def calculate_forecast_period(date_from_ts, date_to_ts, frequency, period):
    if frequency == 'H':
        return int((date_to_ts - date_from_ts).total_seconds() / 3600) + period + 1
    elif frequency == 'D':
        return (date_to_ts - date_from_ts).days + period + 1
    elif frequency == 'W':
        return ((date_to_ts - date_from_ts).days + 1) // 7 + period + 1
    elif frequency == 'M':
        return (date_to_ts.year - date_from_ts.year) * 12 + date_to_ts.month - date_from_ts.month + period + 1
    elif frequency == 'Y':
        return date_to_ts.year - date_from_ts.year + period + 1

def get_actual_values(data, date_from_ts, frequency, period):
    offsets = {
        'H': pd.DateOffset(hours=period),
        'D': pd.DateOffset(days=period),
        'W': pd.DateOffset(weeks=period),
        'M': pd.DateOffset(months=period),
        'Y': pd.DateOffset(years=period)
    }
    return data.loc[date_from_ts:date_from_ts + offsets[frequency]]

def create_result_json(forecast,actual):
    result = []
    for i in range(len(actual)):
        json = {'point_value': actual.iloc[i], 'point_timestamp': actual.index[i], 'forecast': forecast.iloc[i]}
        result.append(json)
    return result