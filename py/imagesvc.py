from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
import requests
import os
import uuid
import base64
import shutil
from typing import Optional
from pathlib import Path
import tempfile

# -----------------------------
# Embedding backends
# -----------------------------
from py.mobileclip_embedding import (
    load_model_once as load_mobileclip,
    compute_vector as compute_mobileclip,
)

from py.nv_dinov_embedding import compute_vector as compute_nv_dino

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
def startup_event():
    try:
        load_mobileclip()
    except Exception as e:
        print(f"[WARN] MobileCLIP not loaded at startup: {e}")

# -----------------------------
# Request Models
# -----------------------------
class ImageRequest(BaseModel):
    image_url: str


class ImageBase64Request(BaseModel):
    image_base64: str
    filename: Optional[str] = None


# -----------------------------
# Helpers
# -----------------------------
def _infer_extension(name_or_url: Optional[str]) -> str:
    if name_or_url and "." in name_or_url:
        ext = name_or_url.split(".")[-1].lower()
        if 1 <= len(ext) <= 4:
            return ext
    return "jpg"


def _save_image_bytes(image_bytes: bytes, ext: str) -> str:
    filename = f"{uuid.uuid4()}.{ext}"
    path = os.path.join(DOWNLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(image_bytes)
    return path


def _compute_vector_from_bytes(
    image_bytes: bytes,
    description: str,
    backend: str,
):
    """
    Dispatch embedding computation based on backend.
    """

    # NV-DINO needs a file path
    if backend == "nv_dino":
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            return compute_nv_dino(tmp_path, description=description)
        finally:
            os.remove(tmp_path)

    # MobileCLIP supports file paths
    elif backend == "mobileclip":
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            return compute_mobileclip(tmp_path, description=description)
        finally:
            os.remove(tmp_path)

    else:
        raise HTTPException(status_code=400, detail="Invalid backend")


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/vectorize-image")
def vectorize_image_url(
    req: ImageRequest,
    backend: str = Query("mobileclip", enum=["mobileclip", "nv_dino"]),
):
    url = req.image_url

    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL")

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        image_bytes = resp.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")

    vector = _compute_vector_from_bytes(
        image_bytes=image_bytes,
        description=url,
        backend=backend,
    )

    ext = _infer_extension(url)
    saved_path = _save_image_bytes(image_bytes, ext)

    return {
        "status": "success",
        "backend": backend,
        "saved_as": os.path.basename(saved_path),
        "path": saved_path,
        "source_url": url,
        "vector": vector,
    }


@app.post("/vectorize-image-base64")
def vectorize_image_base64(
    req: ImageBase64Request,
    backend: str = Query("mobileclip", enum=["mobileclip", "nv_dino"]),
):
    try:
        base64_data = req.image_base64.split(",")[-1]
        image_bytes = base64.b64decode(base64_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

    vector = _compute_vector_from_bytes(
        image_bytes=image_bytes,
        description=req.filename or "base64_image",
        backend=backend,
    )

    ext = _infer_extension(req.filename)
    saved_path = _save_image_bytes(image_bytes, ext)

    return {
        "status": "success",
        "backend": backend,
        "saved_as": os.path.basename(saved_path),
        "path": saved_path,
        "vector": vector,
    }


@app.post("/vectorize-image-upload")
def vectorize_image_upload(
    file: UploadFile = File(...),
    backend: str = Query("mobileclip", enum=["mobileclip", "nv_dino"]),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    image_bytes = file.file.read()
    file.file.close()

    vector = _compute_vector_from_bytes(
        image_bytes=image_bytes,
        description=file.filename,
        backend=backend,
    )

    ext = _infer_extension(file.filename)
    saved_path = _save_image_bytes(image_bytes, ext)

    return {
        "status": "success",
        "backend": backend,
        "original_filename": file.filename,
        "saved_as": os.path.basename(saved_path),
        "path": saved_path,
        "vector": vector,
    }
