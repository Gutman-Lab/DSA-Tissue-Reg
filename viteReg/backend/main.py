"""
FastAPI backend for DSA Tissue Registration application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api import cases, registration, images, annotations, visualization, image_matching_webui
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown tasks"""
    # Startup
    print("Starting FastAPI backend...")
    yield
    # Shutdown
    print("Shutting down FastAPI backend...")


app = FastAPI(
    title="DSA Tissue Registration API",
    description="API for tissue image registration and management",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(cases.router, prefix="/api/cases", tags=["cases"])
app.include_router(registration.router, prefix="/api/registration", tags=["registration"])
app.include_router(images.router, prefix="/api/images", tags=["images"])
app.include_router(annotations.router, prefix="/api/annotations", tags=["annotations"])
app.include_router(visualization.router, prefix="/api/visualization", tags=["visualization"])
app.include_router(image_matching_webui.router, prefix="/api/image-matching", tags=["image-matching"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "DSA Tissue Registration API", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

