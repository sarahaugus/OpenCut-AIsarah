"""Assembly route — upload clips + config, get assembled video."""

import json
import logging
import os
import uuid
from pathlib import Path

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


@router.post("")
async def assemble(
    files: list[UploadFile] = File(..., description="Video/audio files to assemble"),
    config: str = Form(..., description="JSON timeline configuration"),
):
    """Upload video/audio clips and a timeline config, receive a download URL.

    Files are saved temporarily, assembled via FFmpeg with all specified
    effects, transitions, audio overlay, and text overlays, then the
    output is served for download.
    """
    # Parse config
    try:
        timeline_config = json.loads(config)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in config field")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} files allowed")

    # Save uploaded files to temp
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    file_map: dict[int, str] = {}
    saved_paths: list[str] = []

    try:
        for idx, f in enumerate(files):
            if not f.filename:
                continue
            ext = Path(f.filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported format '{ext}' for file '{f.filename}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
                )
            contents = await f.read()
            if len(contents) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File '{f.filename}' too large. Max {MAX_FILE_SIZE // (1024 * 1024)} MB",
                )
            fid = uuid.uuid4().hex[:12]
            fpath = os.path.join(TEMP_UPLOAD_DIR, f"{fid}{ext}")
            with open(fpath, "wb") as out:
                out.write(contents)
            file_map[idx] = fpath
            saved_paths.append(fpath)

        # Build the output
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
