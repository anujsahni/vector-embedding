from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
import requests
import os
import uuid
import base64
import shutil
from typing import Optional

# -----------------------------
# Import both embedding backends
# -----------------------------
from py.mobileclip_embedding import load_model_once as load_mobileclip, compute_vector as compute_mobileclip
from py.nv_dinov_embedding import compute_vector as compute_nv_dino

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

@app.on_event("startup")
def startup_event():
    try:
        load_mobileclip()  # MobileCLIP only, NV-DINO is stateless
    except Exception as e:
        print(f"[WARN] Could not load MobileCLIP model on startup: {e}")

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
    filepath = os.path.join(DOWNLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return filepath

# -----------------------------
# Unified embedding caller
# -----------------------------
def compute_vector_from_bytes(image_bytes: bytes, description="image", backend="mobileclip"):
    """
    Compute embedding from raw bytes using either MobileCLIP or NV-DINO.
    """
    # Save temp file only if NV-DINO requires path
    if backend == "nv_dino":
        # NV-DINO expects a file path
        temp_path = _save_image_bytes(image_bytes, "jpg")
        result = compute_nv_dino(temp_path, description=description)
        # Optionally delete temp file if you want
        # os.remove(temp_path)
        return result
    else:
        # MobileCLIP can work from in-memory bytes
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        # MobileCLIP compute_vector expects path, so save temp file
        temp_path = _save_image_bytes(image_bytes, "jpg")
        result = compute_mobileclip(temp_path, description=description)
        return result

# -----------------------------
# Endpoints
# -----------------------------
@app.post("/vectorize-image")
def vectorize_image_url(req: ImageRequest, backend: str = Query("mobileclip", enum=["mobileclip", "nv_dino"])):
    url = req.image_url
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL")

    ext = _infer_extension(url)
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        image_bytes = resp.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")

    vector_response = compute_vector_from_bytes(image_bytes, description=url, backend=backend)
    filepath = _save_image_bytes(image_bytes, ext)

    return {
        "status": "success",
        "saved_as": os.path.basename(filepath),
        "path": filepath,
        "source_url": url,
        "vector": vector_response
    }

@app.post("/vectorize-image-base64")
def vectorize_image_base64(req: ImageBase64Request, backend: str = Query("mobileclip", enum=["mobileclip", "nv_dino"])):
    try:
        base64_data = req.image_base64.split(",")[-1]
        image_bytes = base64.b64decode(base64_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    vector_response = compute_vector_from_bytes(image_bytes, description=req.filename or "base64_image", backend=backend)
    ext = _infer_extension(req.filename)
    filepath = _save_image_bytes(image_bytes, ext)

    return {
        "status": "success",
        "saved_as": os.path.basename(filepath),
        "path": filepath,
        "vector": vector_response
    }

@app.post("/vectorize-image-upload")
def vectorize_image_upload(file: UploadFile = File(...), backend: str = Query("mobileclip", enum=["mobileclip", "nv_dino"])):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    ext = _infer_extension(file.filename)
    image_bytes = file.file.read()
    file.file.close()

    vector_response = compute_vector_from_bytes(image_bytes, description=file.filename, backend=backend)
    filepath = _save_image_bytes(image_bytes, ext)

    return {
        "status": "success",
        "original_filename": file.filename,
        "saved_as": os.path.basename(filepath),
        "path": filepath,
        "vector": vector_response
    }
