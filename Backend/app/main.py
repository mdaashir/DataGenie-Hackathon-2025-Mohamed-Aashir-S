from fastapi import FastAPI
from Backend.Routes import *

app = FastAPI()

app.include_router(router)
