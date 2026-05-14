"""
Postgres persistence for jobs.

Schema is two normalised tables:

    jobs      — one row per upload, holds status / progress / timing / params
    artifacts — many rows per job, each (job_id, key) → url + optional path

Connection is via psycopg v3's ConnectionPool. The pool is created lazily so
import order doesn't force a DB hit at module-load time (lets tests / linters
import without a running Postgres).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool


DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://badminton:badminton@127.0.0.1:5432/badminton",
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'queued',
    message         TEXT NOT NULL DEFAULT '',
    progress        DOUBLE PRECISION NOT NULL DEFAULT 0,
    total_frames    INTEGER NOT NULL DEFAULT 0,
    frames_done     INTEGER NOT NULL DEFAULT 0,
    started_at      DOUBLE PRECISION NOT NULL,
    finished_at     DOUBLE PRECISION,
    error           TEXT,
    input_filename  TEXT NOT NULL,
    input_path      TEXT NOT NULL,
    result_dir      TEXT NOT NULL,
    infer_url       TEXT NOT NULL,
    conf            DOUBLE PRECISION NOT NULL DEFAULT 0.4,
    iou             DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    do_shuttle      BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS artifacts (
    job_id      TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    key         TEXT NOT NULL,
    url         TEXT NOT NULL,
    path        TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (job_id, key)
);

CREATE INDEX IF NOT EXISTS jobs_created_at_idx ON jobs (created_at DESC);
"""


_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ConnectionPool(
                    conninfo=DATABASE_URL,
                    min_size=1,
                    max_size=8,
                    kwargs={"autocommit": True},
                    open=False,
                )
                _pool.open(wait=True, timeout=30.0)
    return _pool


def init_schema(retries: int = 10, retry_delay: float = 1.5) -> None:
    """Create tables if missing. Retries while Postgres is still booting."""
    last_exc: Optional[Exception] = None
    for _ in range(retries):
        try:
            pool = get_pool()
            with pool.connection() as conn:
                conn.execute(SCHEMA_SQL)
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(retry_delay)
    raise RuntimeError(f"could not initialise DB schema: {last_exc}")


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------
def insert_job(row: dict[str, Any]) -> None:
    cols = [
        "id", "status", "message", "progress",
        "total_frames", "frames_done",
        "started_at", "finished_at", "error",
        "input_filename", "input_path", "result_dir", "infer_url",
        "conf", "iou", "do_shuttle",
    ]
    placeholders = ", ".join(["%s"] * len(cols))
    column_list = ", ".join(cols)
    with get_pool().connection() as conn:
        conn.execute(
            f"INSERT INTO jobs ({column_list}) VALUES ({placeholders})",
            [row.get(c) for c in cols],
        )


def update_job(job_id: str, fields: dict[str, Any]) -> None:
    if not fields:
        return
    keys = list(fields.keys())
    assignments = sql.SQL(", ").join(
        sql.SQL("{} = %s").format(sql.Identifier(k)) for k in keys
    )
    stmt = sql.SQL("UPDATE jobs SET {assignments} WHERE id = %s").format(
        assignments=assignments
    )
    with get_pool().connection() as conn:
        conn.execute(stmt, [fields[k] for k in keys] + [job_id])


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    with get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
            return cur.fetchone()


def upsert_artifact(job_id: str, key: str, url: str, path: Optional[str] = None) -> None:
    with get_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO artifacts (job_id, key, url, path)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (job_id, key) DO UPDATE
              SET url = EXCLUDED.url, path = EXCLUDED.path
            """,
            (job_id, key, url, path),
        )


def list_artifacts(job_id: str) -> dict[str, str]:
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT key, url FROM artifacts WHERE job_id = %s",
                (job_id,),
            )
            return {k: u for k, u in cur.fetchall()}


# Jsonb is re-exported in case callers want to pass JSON payloads later
__all__ = [
    "DATABASE_URL",
    "Jsonb",
    "get_job",
    "get_pool",
    "init_schema",
    "insert_job",
    "list_artifacts",
    "update_job",
    "upsert_artifact",
]
