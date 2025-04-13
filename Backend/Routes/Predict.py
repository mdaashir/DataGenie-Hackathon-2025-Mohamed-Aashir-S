from fastapi import status
from fastapi import APIRouter
from fastapi.responses import JSONResponse

predict_router = APIRouter()
@predict_router.get("/predict")
def predict():
    return JSONResponse(content={"status": "ready"}, status_code=status.HTTP_200_OK)