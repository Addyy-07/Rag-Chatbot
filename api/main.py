"""
api/main.py
───────────
FastAPI application entrypoint.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from api.database.connection import connect_to_mongo, close_mongo_connection
from api.routes import auth_router, user_router, chat_history_router, usage_router, billing_router, webhook_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_to_mongo()
    yield
    # Shutdown
    await close_mongo_connection()

def create_app() -> FastAPI:
    app = FastAPI(
        title="DocChat AI Auth Service",
        description="FastAPI service for authentication and user management.",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Configure CORS to allow Streamlit frontend to make requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For production, restrict this to Streamlit's URL
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/health")
    async def health_check():
        return {"status": "ok"}
    
    # Include routers
    app.include_router(auth_router.router, prefix="/api/v1")
    app.include_router(user_router.router, prefix="/api/v1")
    app.include_router(chat_history_router.router, prefix="/api/v1")
    app.include_router(usage_router.router, prefix="/api/v1")
    app.include_router(billing_router.router, prefix="/api/v1")
    app.include_router(webhook_router.router, prefix="/api/v1")
    
    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
