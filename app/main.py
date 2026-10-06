from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import models  # Register the tables before creating them.
from .database import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield
    engine.dispose()


app = FastAPI(title="Geospatial File Measurement API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}
