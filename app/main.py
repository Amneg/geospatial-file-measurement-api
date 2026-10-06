from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import models
from .database import Base, engine, get_db
from .geo import extract_shapefile, read_features


UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
MAX_UPLOAD_SIZE = 20 * 1024 * 1024


class FileResponse(BaseModel):
    id: int
    filename: str
    feature_count: int
    crs: str | None
    status: str
    uploaded_at: datetime


class FeatureResponse(BaseModel):
    id: int
    feature_id: str | None
    geometry_type: str
    geometry: dict | None
    crs: str | None
    properties: dict
    area_m2: float | None = Field(description="Polygon area in square meters; otherwise null.")
    length_m: float | None = Field(description="LineString length in meters; otherwise null.")
    measurement_status: str | None


class MeasurementsResponse(BaseModel):
    file_id: int
    feature_count: int
    features: list[FeatureResponse]


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield
    engine.dispose()


app = FastAPI(title="Geospatial File Measurement API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


def file_metadata(file_row: models.File):
    return {
        "id": file_row.id,
        "filename": file_row.filename,
        "feature_count": file_row.feature_count,
        "crs": file_row.source_crs,
        "status": file_row.status,
        # SQLite drops timezone information; the stored timestamps are always UTC.
        "uploaded_at": file_row.uploaded_at.replace(tzinfo=timezone.utc),
    }


def save_upload(file: UploadFile, destination: Path):
    size = 0
    with destination.open("wb") as output:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_SIZE:
                raise HTTPException(status_code=413, detail="Upload exceeds the 20 MB limit.")
            output.write(chunk)
    if size == 0:
        raise ValueError("The uploaded file is empty.")


@app.post("/api/files/", status_code=201, response_model=FileResponse)
def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = Path((file.filename or "").replace("\\", "/")).name
    extension = Path(filename).suffix.lower()
    if extension not in {".kml", ".zip"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload .kml or .zip.")

    # Never use a client-supplied filename as a storage path.
    saved_path = UPLOAD_DIR / f"{uuid4().hex}{extension}"
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        save_upload(file, saved_path)
        with TemporaryDirectory() as temporary_directory:
            dataset_path = saved_path
            if extension == ".zip":
                dataset_path = extract_shapefile(saved_path, Path(temporary_directory))
            source_crs, features = read_features(dataset_path)

        file_row = models.File(
            filename=filename,
            source_crs=source_crs,
            feature_count=len(features),
            status="COMPLETED",
        )
        db.add(file_row)
        db.flush()  # SQLite generates the ID needed by the feature rows.
        for feature in features:
            db.add(models.Feature(file_id=file_row.id, **feature))

        # Link the saved original to its database ID without adding another column.
        saved_path = saved_path.replace(UPLOAD_DIR / f"{file_row.id}{extension}")
        response = file_metadata(file_row)
        db.commit()
        return response
    except Exception as error:
        db.rollback()
        saved_path.unlink(missing_ok=True)
        if isinstance(error, HTTPException):
            raise
        if isinstance(error, ValueError):
            raise HTTPException(status_code=400, detail=str(error)) from error
        logging.exception("Could not save upload and feature data")
        raise HTTPException(status_code=500, detail="Could not save upload and feature data.") from error
    finally:
        file.file.close()


@app.get("/api/files/{id}/", response_model=FileResponse)
def get_file(id: int, db: Session = Depends(get_db)):
    file_row = db.get(models.File, id)
    if file_row is None:
        raise HTTPException(status_code=404, detail="File not found.")
    return file_metadata(file_row)


@app.get("/api/files/{id}/measurements/", response_model=MeasurementsResponse)
def get_measurements(id: int, db: Session = Depends(get_db)):
    file_row = db.get(models.File, id)
    if file_row is None:
        raise HTTPException(status_code=404, detail="File not found.")

    features = (
        db.query(models.Feature)
        .filter(models.Feature.file_id == id)
        .order_by(models.Feature.id)
        .all()
    )
    return {
        "file_id": file_row.id,
        "feature_count": len(features),
        "features": [
            {
                "id": feature.id,
                "feature_id": feature.source_feature_id,
                "geometry_type": feature.geometry_type,
                "geometry": feature.geometry,
                "crs": file_row.source_crs,
                "properties": feature.properties,
                "area_m2": feature.area_m2,
                "length_m": feature.length_m,
                "measurement_status": feature.measurement_status,
            }
            for feature in features
        ],
    }
