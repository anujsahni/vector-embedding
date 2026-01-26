import torch
from PIL import Image
import io
from pathlib import Path
from torchvision import transforms
import warnings
import time

warnings.filterwarnings("ignore")

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = str(Path(__file__).parent.parent / "mobileclip_s1_cpu.pt")
DEVICE = "cpu"
BATCH_SIZE = 8  # adjust 4–16 depending on RAM

# Torch thread control
torch.set_grad_enabled(False)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

# -----------------------------
# Globals
# -----------------------------
_model = None

_preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# -----------------------------
# Model loader
# -----------------------------
def load_model_once():
    """Load and warmup the model (call once at app startup)"""
    global _model
    if _model is None:
        print("[INFO] Loading TorchScript MobileCLIP-S1 model...")
        model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
        model.eval()
        model = model.to(memory_format=torch.channels_last)

        # Warmup
        dummy = torch.randn(1, 3, 224, 224).to(
            DEVICE, memory_format=torch.channels_last
        )
        with torch.inference_mode():
            model.encode_image(dummy)

        _model = model
        print("[INFO] MobileCLIP-S1 model ready!")

# -----------------------------
# Image loaders
# -----------------------------
def _load_image_from_path(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((256, 256), Image.BICUBIC)
        return _preprocess(img)

def _load_image_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((256, 256), Image.BICUBIC)
    return _preprocess(img)

# -----------------------------
# Core embedding runner
# -----------------------------
def _run_embedding(batch_tensor):
    with torch.inference_mode():
        emb = _model.encode_image(batch_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb

# -----------------------------
# Embedding APIs (NV-DINO compatible)
# -----------------------------
def compute_vector(image_paths, description="Input Image"):
    """
    Compute embeddings from file paths (batch aware)
    Returns NV-DINO compatible JSON
    """
    if _model is None:
        load_model_once()

    start_time = time.time()

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    if isinstance(description, str):
        description = [description] * len(image_paths)

    tensors = [_load_image_from_path(p) for p in image_paths]
    batch = torch.stack(tensors).to(DEVICE, memory_format=torch.channels_last)

    metadata = []

    for i in range(0, len(batch), BATCH_SIZE):
        chunk = batch[i:i + BATCH_SIZE]
        emb = _run_embedding(chunk)

        for j in range(emb.shape[0]):
            metadata.append({
                "embedding": emb[j].cpu().tolist(),
                "frame_num": i + j
            })

    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "object": "inference.completion",
        "model": "mobileclip_s1",
        "created": int(time.time()),
        "metadata": metadata,
        "usage": {
            "inference_response_time": elapsed_ms
        },
        "description": description[0] if description else "Input Image"
    }

def compute_vector_from_bytes(image_bytes, description="Input Image"):
    """
    Compute embedding from raw image bytes (single image)
    Returns NV-DINO compatible JSON
    """
    if _model is None:
        load_model_once()

    start_time = time.time()

    tensor = _load_image_from_bytes(image_bytes).unsqueeze(0).to(
        DEVICE, memory_format=torch.channels_last
    )

    emb = _run_embedding(tensor)

    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "object": "inference.completion",
        "model": "mobileclip_s1",
        "created": int(time.time()),
        "metadata": [
            {
                "embedding": emb[0].cpu().tolist(),
                "frame_num": 0
            }
        ],
        "usage": {
            "inference_response_time": elapsed_ms
        },
        "description": description
    }
