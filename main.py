from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.engine.hf_engine import HFEngine
from src.api.routes import router
from src.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):

    # -------------------------
    # 🚀 Startup Logic
    # -------------------------

    # Initilize engine
    engine = HFEngine(settings.model_name)

    # Store engine in app state so all routes can access it
    # app.state acts like a global container for shared resources
    app.state.engine = engine

    print("✅ Application startup complete. Model is ready.")

    # Yield control → app starts serving requests after this point
    yield

    # -------------------------
    # 🧹 Shutdown Logic
    # -------------------------

    print("🛑 Application shutting down...")


# Create FastAPI app with lifespan handler
# This ensures startup/shutdown logic is automatically managed

app = FastAPI(lifespan=lifespan)

# Register routes
# All endpoints defined in routes.py will be available
app.include_router(router)
