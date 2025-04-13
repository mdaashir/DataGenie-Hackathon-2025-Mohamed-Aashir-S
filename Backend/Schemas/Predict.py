from pydantic import BaseModel
from typing import List

class TimeSeries(BaseModel):
    Date: str
    Value: float

class TimeSeriesData(BaseModel):
    data: List[TimeSeries]
