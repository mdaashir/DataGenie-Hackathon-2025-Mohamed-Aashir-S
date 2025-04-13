from fastapi import status
from fastapi import APIRouter
from fastapi.responses import JSONResponse

transform_router = APIRouter()
@transform_router.get("/transform")
def transform():
    return JSONResponse(content={"status": "ready"}, status_code=status.HTTP_200_OK)