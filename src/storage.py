from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class NodeRow(Base):
    __tablename__ = "nodes"

    node_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    last_heartbeat: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), default="active")
    last_prediction: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_audio: Mapped[str | None] = mapped_column(String(512), nullable=True)


class UploadRow(Base):
    __tablename__ = "uploads"

    upload_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    path: Mapped[str] = mapped_column(Text)
    node_id: Mapped[str] = mapped_column(String(256))
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    edge_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default="queued")


class JobRow(Base):
    __tablename__ = "pending_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    upload_id: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[float] = mapped_column(Float)
    processed_at: Mapped[float | None] = mapped_column(Float, nullable=True)


class AlertRow(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    upload_id: Mapped[str] = mapped_column(String(64))
    node_id: Mapped[str] = mapped_column(String(256))
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    edge_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_class: Mapped[str] = mapped_column(String(128))
    cloud_confidence: Mapped[float] = mapped_column(Float)
    probabilities_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[float] = mapped_column(Float)
    inference_backend: Mapped[str] = mapped_column(String(32), default="pytorch")


class LocalStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", future=True)
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Session:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def upsert_node(self, node_id: str, lat: float, lon: float, status: str = "active") -> None:
        now = time.time()
        with self.session() as session:
            node = session.get(NodeRow, node_id)
            if node is None:
                session.add(NodeRow(node_id=node_id, lat=lat, lon=lon, status=status, last_heartbeat=now))
                return
            node.lat = lat
            node.lon = lon
            node.status = status
            node.last_heartbeat = max(node.last_heartbeat, now)

    def update_heartbeat(self, node_id: str, lat: float, lon: float, status: str = "active", timestamp: float | None = None) -> None:
        now = time.time() if timestamp is None else timestamp
        with self.session() as session:
            node = session.get(NodeRow, node_id)
            if node is None:
                session.add(NodeRow(node_id=node_id, lat=lat, lon=lon, status=status, last_heartbeat=now))
                return
            node.lat = lat
            node.lon = lon
            node.status = status
            node.last_heartbeat = now

    def update_node_prediction(self, node_id: str, prediction: int, confidence: float, audio_name: str | None) -> None:
        with self.session() as session:
            node = session.get(NodeRow, node_id)
            if node is None:
                return
            node.last_prediction = prediction
            node.last_confidence = confidence
            node.last_audio = audio_name

    def store_upload(
        self,
        upload_id: str,
        path: Path,
        node_id: str,
        lat: float,
        lon: float,
        edge_confidence: float | None,
        filename: str | None,
        created_at: float,
    ) -> None:
        with self.session() as session:
            session.add(
                UploadRow(
                    upload_id=upload_id,
                    path=str(path),
                    node_id=node_id,
                    lat=lat,
                    lon=lon,
                    edge_confidence=edge_confidence,
                    filename=filename,
                    created_at=created_at,
                    status="queued",
                )
            )

    def get_upload(self, upload_id: str) -> UploadRow | None:
        with self.session() as session:
            return session.get(UploadRow, upload_id)

    def mark_upload_processed(self, upload_id: str) -> None:
        with self.session() as session:
            upload = session.get(UploadRow, upload_id)
            if upload is not None:
                upload.status = "processed"

    def enqueue_job(self, upload_id: str, created_at: float | None = None) -> None:
        now = time.time() if created_at is None else created_at
        with self.session() as session:
            existing = session.scalar(select(JobRow).where(JobRow.upload_id == upload_id))
            if existing is None:
                session.add(JobRow(upload_id=upload_id, created_at=now, processed_at=None))

    def claim_next_job(self) -> JobRow | None:
        with self.session() as session:
            job = session.scalar(select(JobRow).where(JobRow.processed_at.is_(None)).order_by(JobRow.created_at.asc(), JobRow.id.asc()))
            if job is None:
                return None
            job.processed_at = time.time()
            session.flush()
            return JobRow(id=job.id, upload_id=job.upload_id, created_at=job.created_at, processed_at=job.processed_at)

    def record_alert(
        self,
        upload_id: str,
        node_id: str,
        lat: float,
        lon: float,
        edge_confidence: float | None,
        predicted_class: str,
        cloud_confidence: float,
        probabilities: dict[str, float],
        inference_backend: str,
        created_at: float | None = None,
    ) -> None:
        now = time.time() if created_at is None else created_at
        with self.session() as session:
            session.add(
                AlertRow(
                    upload_id=upload_id,
                    node_id=node_id,
                    lat=lat,
                    lon=lon,
                    edge_confidence=edge_confidence,
                    predicted_class=predicted_class,
                    cloud_confidence=cloud_confidence,
                    probabilities_json=json.dumps(probabilities),
                    created_at=now,
                    inference_backend=inference_backend,
                )
            )

    def list_nodes(self) -> list[NodeRow]:
        with self.session() as session:
            return list(session.scalars(select(NodeRow).order_by(NodeRow.node_id)).all())

    def list_recent_alerts(self, limit: int = 20) -> list[AlertRow]:
        with self.session() as session:
            return list(session.scalars(select(AlertRow).order_by(AlertRow.created_at.desc()).limit(limit)).all())

    def list_uploads(self, limit: int = 50) -> list[UploadRow]:
        with self.session() as session:
            return list(session.scalars(select(UploadRow).order_by(UploadRow.created_at.desc()).limit(limit)).all())

    def pending_jobs_count(self) -> int:
        with self.session() as session:
            return int(session.query(JobRow).filter(JobRow.processed_at.is_(None)).count())
