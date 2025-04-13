from fastapi import FastAPI
from Backend.Routes import router

app = FastAPI(title="Timeseries Backend API", version="1.0.0",description="This is the backend API for the Timeseries application.")

app.include_router(router)
