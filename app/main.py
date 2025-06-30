from fastapi import FastAPI
from app.api.v1.routers import router as api_router
from app.core.exception_handlers import register_exception_handlers

app = FastAPI(title="Violation D/api/v1/detection API")

# Đăng ký routers
app.include_router(api_router, prefix="/api/v1")

# Đăng ký exception handlers
register_exception_handlers(app)





