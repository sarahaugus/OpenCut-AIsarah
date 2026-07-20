"""Assembly route — upload clips + config, get assembled video."""

import json
import logging
import os
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.config import settings
from app.services.assembly_service import assembly_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/assembly", tags=["assembly"])

TEMP_UPLOAD_DIR = "/tmp/opencut_uploads"
MAX_FILES = 20
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB
ALLOWED_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
URL_TIMEOUT = 120


def _ext_from_url(url: str, content_type: str) -> str:
    """Guess file extension from URL path or content-type."""
    p = Path(url.split("?")[0]).suffix.lower()
    if p in ALLOWED_EXTENSIONS:
        return p
    if "audio" in content_type:
        return ".mp3"
    if "video" in content_type:
        return ".mp4"
    return ".mp4"


async def _fetch_url(url: str, fid: str) -> str:
    """Download a URL to a temp file and return the path."""
    try:
        async with httpx.AsyncClient(timeout=URL_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            ext = _ext_from_url(url, resp.headers.get("content-type", ""))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {url} — {e}")

    fpath = os.path.join(TEMP_UPLOAD_DIR, f"{fid}{ext}")
    with open(fpath, "wb") as out:
        out.write(resp.content)
    return fpath


@router.post("")
async def assemble(
    files: list[UploadFile] | None = File(default=None, description="Video/audio files to upload"),
    config: str = Form(..., description="JSON timeline configuration"),
):
    """Assemble video from uploaded files and/or remote URLs.

    Each clip in config can specify either `file_index` (referring to an uploaded file
    by upload order) or `url` (the backend fetches the file from a remote URL).
    The same applies to `audio_overlay.file_index` / `audio_overlay.url`.
    """
    try:
        timeline_config = json.loads(config)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in config field")

    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    file_map: dict[int, str] = {}
    saved_paths: list[str] = []

    try:
        # Process uploaded files
        next_idx = 0
        if files:
            if len(files) > MAX_FILES:
                raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} files allowed")
            for f in files:
                if not f.filename:
                    continue
                fext = Path(f.filename).suffix.lower()
                if fext not in ALLOWED_EXTENSIONS:
                    raise HTTPException(status_code=400, detail=f"Unsupported format for '{f.filename}'")
                contents = await f.read()
                if len(contents) > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail=f"File too large: {f.filename}")
                fid = uuid.uuid4().hex[:12]
                fpath = os.path.join(TEMP_UPLOAD_DIR, f"{fid}{fext}")
                with open(fpath, "wb") as out:
                    out.write(contents)
                file_map[next_idx] = fpath
                saved_paths.append(fpath)
                next_idx += 1

        # Process remote URLs from config
        clips = timeline_config.get("timeline", timeline_config).get("clips", [])
        url_entries: list[tuple[int, str]] = []

        for ci, clip in enumerate(clips):
            if "url" in clip:
                url_entries.append((next_idx, clip["url"]))
                clip["file_index"] = next_idx
                next_idx += 1

        ao = timeline_config.get("timeline", timeline_config).get("audio_overlay", {})
        if ao and "url" in ao:
            url_entries.append((next_idx, ao["url"]))
            ao["file_index"] = next_idx
            next_idx += 1

        for idx, url in url_entries:
            fid = uuid.uuid4().hex[:12]
            fpath = await _fetch_url(url, fid)
            file_map[idx] = fpath
            saved_paths.append(fpath)

        if not file_map:
            raise HTTPException(status_code=400, detail="No files provided — upload or specify url in config")

        output_path = await assembly_service.assemble(timeline_config, file_map)

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Assembly produced no output file")

        filename = os.path.basename(output_path)
        download_url = f"{settings.ASSEMBLY_DOWNLOAD_BASE_URL}/{filename}"

        return {"url": download_url}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in saved_paths:
            try:
                os.remove(p)
            except OSError:
                pass


@router.get("/download/{filename}")
async def download_assembly(filename: str):
    """Download an assembled video by filename."""
    safe = Path(filename).name  # prevent path traversal
    filepath = os.path.join("/tmp/opencut_assembly", safe)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found or expired")
    return FileResponse(path=filepath, media_type="video/mp4", filename=safe)
