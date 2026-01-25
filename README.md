
# Image Vectorization Service

This service exposes a REST API to download an image from a provided URL and generate a vector embedding.  
It supports two embedding backends:

1. **MobileCLIP-S1** – local TorchScript model, vector size **512**.
2. **NVIDIA NV-DINOv2** – cloud API, vector size **1536**, requires an NVIDIA API key.

The service is built with **FastAPI** and can run locally or on an EC2 instance.

## Features

- Accepts an **image URL** as input.
- Accepts an **image in Base64 JSON** as input.
- Accepts an **image file via multipart upload** (ideal for mobile cameras).
- Downloads the image locally (where the service is running).
- Generates a **vector embedding** from the selected backend.
- Returns JSON response with the vector, saved filename, source URL, and backend info.

---

## Backend Selection

You can select the backend by adding the query parameter `?backend=<model>` to your request:  

| Backend       | Vector Dimension | Notes                                           |
|---------------|----------------|------------------------------------------------|
| `mobileclip`  | 512            | Uses local MobileCLIP-S1 TorchScript model    |
| `nv_dino`     | 1536           | Uses NVIDIA NV-DINOv2 cloud API, requires NVIDIA_API_KEY |

If no backend is specified, `mobileclip` is used by default.

---

## Prerequisites

- Python 3.9+
- `pip` packages: `fastapi`, `uvicorn`, `requests`, `pydantic`, `python-multipart`
- Optional: NVIDIA API key for NV-DINOv2 embedding:

```bash
export NVIDIA_API_KEY="your_actual_api_key_here"
````

---

## How to Run the Service

The service accepts images via URL, Base64, or file upload and returns a vector embedding.
The service listens on port **8000** by default.

### Start the Service Locally

```bash
uvicorn py.imagesvc:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs:

```bash
http://localhost:8000/docs
```

---

## Example Requests

### 1. Vectorize Image by URL

Default backend (MobileCLIP-S1):

```bash
curl -X POST "http://<hostname>:8000/vectorize-image" \
  -H "Content-Type: application/json" \
  -d '{
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/e/e4/Morgan_Freeman_Deauville_2018.jpg"
      }'
```

Specify NV-DINOv2 backend:

```bash
curl -X POST "http://<hostname>:8000/vectorize-image?backend=nv_dino" \
  -H "Content-Type: application/json" \
  -d '{
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/e/e4/Morgan_Freeman_Deauville_2018.jpg"
      }'
```

### 2. Vectorize Image by Base64

Convert image to Base64 (macOS/Linux):

```bash
base64 image.png > img.b64
```

Call the service with default backend:

```bash
curl -X POST "http://<hostname>:8000/vectorize-image-base64" \
  -H "Content-Type: application/json" \
  -d "{
        \"image_base64\": \"$(cat img.b64)\",
        \"filename\": \"image.png\"
      }"
```

Specify NV-DINOv2 backend:

```bash
curl -X POST "http://<hostname>:8000/vectorize-image-base64?backend=nv_dino" \
  -H "Content-Type: application/json" \
  -d "{
        \"image_base64\": \"$(cat img.b64)\",
        \"filename\": \"image.png\"
      }"
```

### 3. Vectorize Image via Multipart Upload

```bash
curl -X POST "http://<hostname>:8000/vectorize-image-upload" \
  -F "file=@image.png"
```

Specify NV-DINOv2 backend:

```bash
curl -X POST "http://<hostname>:8000/vectorize-image-upload?backend=nv_dino" \
  -F "file=@image.png"
```

---

## Downloads Folder

All images uploaded via URL, Base64, or multipart are saved in the `downloads/` folder.

---

## ⚙️ Systemd Service Setup

Run the service automatically at startup.

### 1. Create the service file

```bash
sudo vi /etc/systemd/system/imagesvc.service
```

### 2. Add the following content

```ini
[Unit]
Description=FastAPI Image Service
After=network.target

[Service]
User=ec2-user
Group=ec2-user
WorkingDirectory=/home/ec2-user/demo/vector-embedding
Environment="NVIDIA_API_KEY=nvapi-****"
ExecStart=/usr/bin/python3 -m uvicorn py.imagesvc:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

> Replace `WorkingDirectory` and `Environment` with your actual paths and NVIDIA API key.

### 3. Reload systemd

```bash
sudo systemctl daemon-reload
```

### 4. Enable the service at boot

```bash
sudo systemctl enable imagesvc.service
```

### 5. Start the service

```bash
sudo systemctl start imagesvc.service
```

### 6. Check the service status

```bash
sudo systemctl status imagesvc.service
```



Now your FastAPI Image Service supports **both MobileCLIP-S1 (512)** and **NV-DINOv2 (1536)** backends with a simple `?backend=<model>` parameter.
