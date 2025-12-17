# Image Vectorization Service

This service exposes a REST API to download an image from a provided URL and generate a vector embedding using NVIDIA's **NV-DINOv2** model. The service is built with **FastAPI** and is designed to run on an EC2 instance.

## Features

- Accepts an **image URL** as input.
- Accepts an **image in Base64 JSON** as input.
- Accepts an **image file via multipart upload** (ideal for mobile cameras).
- Downloads the image locally (where the service is running).
- Generates a **vector of size 1536** using NVIDIA NV-DINOv2 embeddings.
- Returns JSON response with the vector, saved filename, and source URL.

## Prerequisites

- Python 3.9+ installed
- `pip` packages: `fastapi`, `uvicorn`, `requests`, `pydantic`, `python-multipart`
- A valid NVIDIA API key for NV-DINOv2 embedding.

The NVIDIA API key must be set in the environment:

```bash
export NVIDIA_API_KEY="your_actual_api_key_here"
```

## How to Run the Image Embedding Service

The Image Embedding Service accepts an image URL and returns an 1536 dimensional vector embedding generated using NVIDIA's `nv-dinov2` model.  
The service listens on port **8000** by default.

### Start the Service Locally

```bash
uvicorn py.imagesvc:app --host 0.0.0.0 --port 8000 --reload
```

The API can be explored interactively at:

```bash
http://localhost:8000/docs

```

### Example cURL Request

#### 1. Vectorize Image by URL

```bash
curl -X POST http://<hostname>:8000/vectorize-image \
  -H "Content-Type: application/json" \
  -d '{
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/e/e4/Morgan_Freeman_Deauville_2018.jpg"
      }'
```
Replace:

- `<hostname>` with your public hostname or public IP
- `image_url` with any valid image URL

#### 2. Vectorize Image by Base64

Convert image to Base64 (macOS/Linux):
```bash
base64 image.png > img.b64
```

Call the service:

```bash
curl -X POST http://<hostname>:8000/vectorize-image-base64 \
  -H "Content-Type: application/json" \
  -d "{
        \"image_base64\": \"$(cat img.b64)\",
        \"filename\": \"image.png\"
      }"
```

#### 3. Vectorize Image via Multipart Upload

```bash
curl -X POST http://<hostname>:8000/vectorize-image-upload \
  -F "file=@image.png"
```

This endpoint streams the image bytes directly to the server, saves the file in the downloads/ folder, and returns the vector embedding. Ideal for Android or mobile camera uploads.

### Downloads Folder

All uploaded images via URL, Base64, or multipart are saved in the downloads/ folder.

## ⚙️ Systemd Service Setup

To run the Image Service automatically at system startup, you can create a `systemd` service.

### 1. Create the service file

```bash
sudo vi /etc/systemd/system/imagesvc.service
```

### 2. Add the following content to the file
```bash
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

Note:

- Replace WorkingDirectory with the path to your project folder.
- Replace Environment with your actual NVIDIA API key.
- Make sure the ExecStart path points to your FastAPI app module.

### 3. Reload systemd to apply the new service
```bash
sudo systemctl daemon-reload
```
### 4. Enable the service to start on boot
```bash
sudo systemctl enable imagesvc.service
```
### 5. Start the service immediately
```bash
sudo systemctl start imagesvc.service
```
### 6. Check the service status
```bash
sudo systemctl status imagesvc.service
```

This setup ensures that your FastAPI Image Service runs automatically and restarts on failure.
