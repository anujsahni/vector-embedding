import os
import sys
import uuid
import base64

import requests

# NVAI endpoint for the NV-DINOv2 NIM
nvai_url="https://ai.api.nvidia.com/v1/cv/nvidia/nv-dinov2"

header_auth = os.getenv('NVIDIA_API_KEY')

def _upload_asset(input_bytes, description):

    # Uploads an asset to the NVCF API and returns the asset_id (UUID)

    assets_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"

    headers = {
        "Authorization": header_auth,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    s3_headers = {
        "x-amz-meta-nvcf-asset-description": description,
        "content-type": "image/jpeg",
    }

    payload = {"contentType": "image/jpeg", "description": description}

    response = requests.post(assets_url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    asset_url = response.json()["uploadUrl"]
    asset_id = response.json()["assetId"]

    response = requests.put(asset_url, data=input_bytes, headers=s3_headers, timeout=300)
    response.raise_for_status()

    return uuid.UUID(asset_id)

def compute_vector(image_path, description="Input Image"):

    # Consolidates uploading and inference. Returns JSON response from NVAI.

    # Read image bytes
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Small images: send as base64
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
            ],
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": header_auth,
            "Accept": "application/json"
        }
    else:
        # Large images: upload as asset
        asset_id = _upload_asset(image_bytes, description)
        payload = {"messages": []}

        asset_list = f"{asset_id}"

        headers = {
            "Content-Type": "application/json",
            "NVCF-INPUT-ASSET-REFERENCES": asset_list,
            "NVCF-FUNCTION-ASSET-IDS": asset_list,
            "Authorization": header_auth,
        }

    # Call NVAI endpoint
    response = requests.post(nvai_url, headers=headers, json=payload)

    # Return JSON response or raise exception
    if response.status_code == 200:
        return response.json()
    else:
        raise RuntimeError(f"Inference failed: {response.status_code} - {response.text}")
