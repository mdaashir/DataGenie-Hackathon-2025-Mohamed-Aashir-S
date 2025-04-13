from fastapi import status
from fastapi import APIRouter
from fastapi.responses import JSONResponse

health_router = APIRouter()
@health_router.get("/health")
def health():
    return JSONResponse(content={"status": "ok"}, status_code=status.HTTP_200_OK)
