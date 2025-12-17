import os
import uuid
import base64
import requests

# -----------------------------
# Config
# -----------------------------
NVAI_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nv-dinov2"
ASSETS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
API_KEY = os.getenv("NVIDIA_API_KEY")
HEADER_AUTH = f"Bearer {API_KEY}"

# -----------------------------
# Private helpers
# -----------------------------
def _upload_asset(input_bytes: bytes, description: str) -> uuid.UUID:
    """
    Uploads an asset to NVCF API and returns the asset UUID.
    """
    headers = {
        "Authorization": HEADER_AUTH,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    s3_headers = {
        "x-amz-meta-nvcf-asset-description": description,
        "content-type": "image/jpeg",
    }

    payload = {"contentType": "image/jpeg", "description": description}

    response = requests.post(ASSETS_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    asset_info = response.json()
    asset_url = asset_info["uploadUrl"]
    asset_id = asset_info["assetId"]

    # Upload bytes to S3
    response = requests.put(asset_url, data=input_bytes, headers=s3_headers, timeout=300)
    response.raise_for_status()

    return uuid.UUID(asset_id)

# -----------------------------
# Embedding functions
# -----------------------------
def compute_vector(image_path: str, description: str = "Input Image"):
    """Compute NV-DINOv2 embedding from local file path"""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return compute_vector_from_bytes(image_bytes, description)

def compute_vector_from_bytes(image_bytes: bytes, description: str = "Input Image"):
    """Compute NV-DINOv2 embedding from raw bytes (upload/base64 handled automatically)"""
    headers = {"Authorization": HEADER_AUTH}

    # Small images: embed as base64
    if len(image_bytes) < 200_000:
        image_b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "messages": [
                {
                    "content": {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    }
                }
            ]
        }
        headers.update({"Content-Type": "application/json", "Accept": "application/json"})
    else:
        # Large images: upload as asset
        asset_id = _upload_asset(image_bytes, description)
        payload = {"messages": []}
        headers.update({
            "Content-Type": "application/json",
            "NVCF-INPUT-ASSET-REFERENCES": str(asset_id),
            "NVCF-FUNCTION-ASSET-IDS": str(asset_id),
        })

    # Call NVAI endpoint
    response = requests.post(NVAI_URL, headers=headers, json=payload, timeout=60)

    if response.status_code == 200:
        result = response.json()
        result["description"] = description
        return result
    else:
        raise RuntimeError(f"NV-DINOv2 inference failed: {response.status_code} - {response.text}")
