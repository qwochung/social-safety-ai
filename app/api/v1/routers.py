from fastapi import APIRouter
from app.api.v1.endpoints import content, image, image_url

router = APIRouter()
router.include_router(content.router, tags=["Content"])
router.include_router(image.router, tags=["Image"])
router.include_router(image_url.router, tags=["ImageUrl"])
