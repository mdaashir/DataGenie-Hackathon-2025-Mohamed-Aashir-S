from fastapi import APIRouter
from Backend.Routes.Home import home_router
from Backend.Routes.Health import health_router
from Backend.Routes.Predict import predict_router

router = APIRouter()

router.include_router(home_router, tags=["Home"])
router.include_router(health_router, tags=["Health"])
router.include_router(predict_router, tags=["Predict"])