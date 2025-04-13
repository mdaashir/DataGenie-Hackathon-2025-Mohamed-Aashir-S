from fastapi import FastAPI
from Backend.Routes import router

app = FastAPI()

app.include_router(router)
