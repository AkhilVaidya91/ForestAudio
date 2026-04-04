from __future__ import annotations

from .storage import LocalStore


class LocalJobQueue:
    def __init__(self, store: LocalStore) -> None:
        self.store = store

    def enqueue(self, upload_id: str, created_at: float | None = None) -> None:
        self.store.enqueue_job(upload_id, created_at=created_at)

    def claim(self) -> str | None:
        job = self.store.claim_next_job()
        if job is None:
            return None
        return job.upload_id

    def size(self) -> int:
        return self.store.pending_jobs_count()
