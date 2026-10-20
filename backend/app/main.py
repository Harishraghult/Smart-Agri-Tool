from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.session import engine, Base
from app.api.v1.endpoints import (
    diagnosis,
    ripeness,
    field,
    advisory,
    crop,
    chat,
    health,
    stats
)

# Initialize database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Standard 422 body, but long rejected inputs are truncated instead of echoed in full."""
    errors = []
    for err in exc.errors():
        err = dict(err)
        if isinstance(err.get("input"), str) and len(err["input"]) > 100:
            err["input"] = err["input"][:100] + "…"
        elif not isinstance(err.get("input"), (str, int, float, bool, type(None))):
            err.pop("input", None)
        errors.append(err)
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for dev / frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers under /api/v1
app.include_router(health.router, prefix=settings.API_V1_STR, tags=["Health"])
app.include_router(stats.router, prefix=settings.API_V1_STR, tags=["Dashboard"])
app.include_router(diagnosis.router, prefix=f"{settings.API_V1_STR}/diagnosis", tags=["Image Diagnosis"])
app.include_router(ripeness.router, prefix=f"{settings.API_V1_STR}/ripeness", tags=["Ripeness & Quality"])
app.include_router(field.router, prefix=f"{settings.API_V1_STR}/field", tags=["Field Intelligence"])
app.include_router(advisory.router, prefix=f"{settings.API_V1_STR}/advisory", tags=["Crop Advisory & Weather"])
app.include_router(crop.router, prefix=f"{settings.API_V1_STR}/crop", tags=["Crop Recommender"])
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat", tags=["Agricultural Chatbot"])

@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/status"
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "project": settings.PROJECT_NAME}
