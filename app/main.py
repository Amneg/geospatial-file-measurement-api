from contextlib import asynccontextmanager
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from . import models
from .database import Base, engine, get_db
from .geo import extract_shapefile, read_features


UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
MAX_UPLOAD_SIZE = 20 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield
    engine.dispose()


app = FastAPI(title="Geospatial File Measurement API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


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


@app.post("/api/files/", status_code=201)
def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = Path((file.filename or "").replace("\\", "/")).name
    extension = Path(filename).suffix.lower()
    if extension not in {".kml", ".zip"}:
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload .kml or .zip.")

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
        response = {
            "id": file_row.id,
            "filename": file_row.filename,
            "source_crs": file_row.source_crs,
            "feature_count": file_row.feature_count,
            "status": file_row.status,
            "uploaded_at": file_row.uploaded_at,
        }
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
