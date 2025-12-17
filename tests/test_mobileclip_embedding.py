# tests/test_mobileclip_embedding.py
from py.mobileclip_embedding import compute_vector
from pathlib import Path

# Path to test image (convert Path -> str)
image_path = Path(__file__).parent / "test.jpg"
result = compute_vector(str(image_path))  # <-- convert to string

embedding = result["data"][0]["embedding"]

print("Model:", result["model"])
print("Embedding length:", len(embedding))
print("First 10 values:", embedding[:10])
