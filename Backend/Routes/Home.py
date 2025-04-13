from fastapi import status
from fastapi import APIRouter
from fastapi.responses import JSONResponse

home_router = APIRouter()

@home_router.get("")
def home():
    return JSONResponse(content={"message":"Hello World"}, status_code=status.HTTP_200_OK)