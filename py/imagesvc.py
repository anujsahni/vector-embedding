from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import uuid

# Import the embedding function
from py.embedding import compute_vector

app = FastAPI()

class ImageRequest(BaseModel):
    image_url: str

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
