from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String

from .database import Base


class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    source_crs = Column(String, nullable=True)
    feature_count = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="pending")
    # SQLite stores this timestamp without an offset; always write UTC.
    uploaded_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class Feature(Base):
    __tablename__ = "features"

    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=False, index=True)
    source_feature_id = Column(String, nullable=True)
    geometry_type = Column(String, nullable=False)
    geometry = Column(JSON, nullable=False)
    properties = Column(JSON, nullable=False, default=dict)
    area_m2 = Column(Float, nullable=True)
    length_m = Column(Float, nullable=True)
    measurement_status = Column(String, nullable=True)
