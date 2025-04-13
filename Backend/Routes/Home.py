from fastapi import APIRouter
from fastapi import status

home_router = APIRouter()

@home_router.get("/")
def home():
    return "Heo World", status.HTTP_200_OK