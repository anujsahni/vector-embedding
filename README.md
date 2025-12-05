# Image Vectorization Service

This service exposes a REST API to download an image from a provided URL and generate a vector embedding using NVIDIA's **NV-DINOv2** model. The service is built with **FastAPI** and is designed to run on an EC2 instance.

---

## Features

- Accepts an **image URL** as input.
- Downloads the image locally (where the service is running).
- Generates a **vector of size 1024** using NVIDIA NV-DINOv2 embeddings.
- Returns JSON response with the vector, saved filename, and source URL.

---

## Prerequisites

- Python 3.9+ installed
- `pip` packages: `fastapi`, `uvicorn`, `requests`, `pydantic`
- A valid NVIDIA API key for NV-DINOv2 embedding.

The NVIDIA API key must be set in the environment:

```bash
export NVIDIA_API_KEY="your_actual_api_key_here"
