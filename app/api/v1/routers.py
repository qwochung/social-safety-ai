from fastapi import APIRouter
from app.api.v1.endpoints import content, image

router = APIRouter()
router.include_router(content.router, tags=["Content"])
router.include_router(image.router, tags=["Image"])
