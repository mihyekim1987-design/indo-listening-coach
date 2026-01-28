# api/jobs.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Dict, Optional
from uuid import uuid4


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus = JobStatus.pending
    result: Optional[Any] = None
    error: Optional[str] = None


_lock = Lock()
_jobs: Dict[str, JobRecord] = {}


def create_job() -> str:
    job_id = str(uuid4())
    with _lock:
        _jobs[job_id] = JobRecord(job_id=job_id)
    return job_id


def set_status(job_id: str, status: JobStatus) -> None:
    with _lock:
        _jobs[job_id].status = status


def set_result(job_id: str, result: Any) -> None:
    with _lock:
        _jobs[job_id].status = JobStatus.done
        _jobs[job_id].result = result
        _jobs[job_id].error = None


def set_error(job_id: str, error: str) -> None:
    with _lock:
        _jobs[job_id].status = JobStatus.failed
        _jobs[job_id].error = error


def get_job(job_id: str) -> JobRecord:
    with _lock:
        if job_id not in _jobs:
            raise KeyError(f"job_id not found: {job_id}")
        return _jobs[job_id]
