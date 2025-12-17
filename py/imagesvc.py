from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import requests
import os
import uuid
import base64
import shutil
from typing import Optional


# Import the embedding function
from py.embedding import compute_vector

app = FastAPI()

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ======================
# Request Models
# ======================

class ImageRequest(BaseModel):
    image_url: str


class ImageBase64Request(BaseModel):
    image_base64: str
    filename: Optional[str] = None


# ======================
# Helper Functions (NEW)
# ======================

def _infer_extension(name_or_url: Optional[str]) -> str:
    if name_or_url and "." in name_or_url:
        ext = name_or_url.split(".")[-1].lower()
        if 1 <= len(ext) <= 4:
            return ext
    return "jpg"


def _save_image_bytes(image_bytes: bytes, ext: str) -> str:
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(DOWNLOAD_DIR, filename)

    try:
        with open(filepath, "wb") as f:
            f.write(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")

    return filepath


def _save_upload_file(upload_file: UploadFile, ext: str) -> str:
    """
    Streams multipart file directly to disk (no full RAM load)
    """
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(DOWNLOAD_DIR, filename)

    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
    finally:
        upload_file.file.close()

    return filepath



@app.post("/vectorize-image")
def download_image(req: ImageRequest):
    url = req.image_url

    # Basic URL validation
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=400, detail="Invalid URL")

    # Create downloads folder
    os.makedirs("downloads", exist_ok=True)

    # Generate unique local filename
    ext = url.split('.')[-1].lower()
    if len(ext) > 4:  # fallback if extension is weird
        ext = "jpg"

    filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join("downloads", filename)

    # Download the file with browser headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/116.0 Safari/537.36"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(resp.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")

     # --- Call embedding logic ---
    vector_response = compute_vector(filepath)

    return {
        "status": "success",
        "saved_as": filename,
        "path": filepath,
        "source_url": url,
        "vector": vector_response
    }


# ======================
# BASE64 ENDPOINT
# ======================

@app.post("/vectorize-image-base64")
def vectorize_image_base64(req: ImageBase64Request):
    try:
        base64_data = req.image_base64
        if "," in base64_data:
            _, base64_data = base64_data.split(",", 1)

        image_bytes = base64.b64decode(base64_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    ext = _infer_extension(req.filename)
    filepath = _save_image_bytes(image_bytes, ext)

    vector_response = compute_vector(filepath)

    return {
        "status": "success",
        "saved_as": os.path.basename(filepath),
        "path": filepath,
        "vector": vector_response
    }


# ======================
# MULTIPART UPLOAD ENDPOINT (NEW)
# ======================

@app.post("/vectorize-image-upload")
def vectorize_image_upload(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")

    ext = _infer_extension(file.filename)
    filepath = _save_upload_file(file, ext)

    vector_response = compute_vector(filepath)

    return {
        "status": "success",
        "original_filename": file.filename,
        "saved_as": os.path.basename(filepath),
        "path": filepath,
        "vector": vector_response
    }
