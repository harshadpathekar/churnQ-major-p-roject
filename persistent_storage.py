"""Persist Vercel runtime files in a private Vercel Blob store."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from flask import current_app, g, make_response

from config import DATA_DIR, INITIAL_PROCESSED_FOLDER, PROCESSED_FOLDER


_BLOB_PREFIX = "churnq/"
_CACHED_ETAGS: dict[str, str] = {}
_CACHED_PATHS: set[str] = set()


def _blob_call(operation):
    """Retry brief Blob service interruptions before failing a request."""
    for attempt in range(3):
        try:
            return operation()
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            if status_code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
            time.sleep(0.25 * (2 ** attempt))


def _blob_sdk():
    if not os.getenv("BLOB_READ_WRITE_TOKEN") and not os.getenv("BLOB_STORE_ID"):
        return None
    from vercel.blob import sync

    return sync


def _local_path(pathname: str) -> Path | None:
    if not pathname.startswith(_BLOB_PREFIX):
        return None
    relative = pathname[len(_BLOB_PREFIX):]
    if not (relative.startswith("uploads/") or relative.startswith("processed/")):
        return None
    root = Path(DATA_DIR).resolve()
    destination = (root / relative).resolve()
    if destination == root or root not in destination.parents:
        return None
    return destination


def _iter_local_files():
    for folder in (Path(PROCESSED_FOLDER), Path(DATA_DIR) / "uploads"):
        if folder.is_dir():
            for path in folder.rglob("*"):
                if path.is_file():
                    yield path


def _pathname(path: Path) -> str:
    relative = path.resolve().relative_to(Path(DATA_DIR).resolve())
    return _BLOB_PREFIX + relative.as_posix()


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_unchanged_bundled_file(path: Path, digest: str) -> bool:
    try:
        relative = path.relative_to(Path(PROCESSED_FOLDER))
    except ValueError:
        return False
    bundled = Path(INITIAL_PROCESSED_FOLDER) / relative
    return bundled.is_file() and _digest(bundled) == digest


def _restore_from_blob() -> None:
    sdk = _blob_sdk()
    if sdk is None:
        return

    remote_blobs = _blob_call(
        lambda: list(sdk.query_blobs(prefix=_BLOB_PREFIX))
    )
    remote_paths = {blob.pathname for blob in remote_blobs}
    for stale_path in _CACHED_PATHS - remote_paths:
        local = _local_path(stale_path)
        if local and local.is_file():
            local.unlink()

    for blob in remote_blobs:
        local = _local_path(blob.pathname)
        if local is None:
            continue
        if local.is_file() and _CACHED_ETAGS.get(blob.pathname) == blob.etag:
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        result = _blob_call(
            lambda: sdk.get_blob(blob.pathname, access="private")
        )
        local.write_bytes(result.body)
        _CACHED_ETAGS[blob.pathname] = blob.etag

    # Seed the store with model files included in the deployment.
    if not remote_paths and os.path.isdir(INITIAL_PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER, exist_ok=True)
        for source in Path(INITIAL_PROCESSED_FOLDER).rglob("*"):
            if source.is_file():
                relative = source.relative_to(Path(INITIAL_PROCESSED_FOLDER))
                destination = Path(PROCESSED_FOLDER) / relative
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(source.read_bytes())

    g._blob_sdk = sdk
    g._blob_remote_paths = remote_paths
    g._blob_snapshot = {
        _pathname(path): _digest(path) for path in _iter_local_files()
    }


def _persist_to_blob(response):
    sdk = getattr(g, "_blob_sdk", None)
    if sdk is None:
        return response

    try:
        remote_paths = set(g._blob_remote_paths)
        snapshot = g._blob_snapshot
        current_paths = set()

        for path in _iter_local_files():
            pathname = _pathname(path)
            current_paths.add(pathname)
            digest = _digest(path)
            if pathname in remote_paths and snapshot.get(pathname) == digest:
                continue
            if (pathname not in remote_paths
                    and _is_unchanged_bundled_file(path, digest)):
                continue
            result = _blob_call(
                lambda: sdk.put_blob(
                    pathname,
                    path.read_bytes(),
                    access="private",
                    allow_overwrite=True,
                )
            )
            _CACHED_ETAGS[pathname] = result.etag

        removed_paths = remote_paths - current_paths
        if removed_paths:
            _blob_call(lambda: sdk.delete_blob(sorted(removed_paths)))
            for pathname in removed_paths:
                _CACHED_ETAGS.pop(pathname, None)

        _CACHED_PATHS.clear()
        _CACHED_PATHS.update(current_paths)
        return response
    except Exception:
        current_app.logger.exception("Could not persist app files to Vercel Blob")
        return make_response(
            "The app could not save this request's files to persistent storage. "
            "Please try again.",
            503,
        )


def register_blob_storage(app) -> None:
    """Sync app working files with private Blob storage on each Vercel request."""

    if os.getenv("VERCEL") != "1":
        return

    @app.before_request
    def restore_persistent_files():
        if not (os.getenv("BLOB_READ_WRITE_TOKEN") or os.getenv("BLOB_STORE_ID")):
            return None
        try:
            _restore_from_blob()
        except Exception:
            app.logger.exception("Could not restore app files from Vercel Blob")
            return make_response(
                "Persistent storage is temporarily unavailable. Please try again.",
                503,
            )
        return None

    @app.after_request
    def persist_response_files(response):
        return _persist_to_blob(response)
