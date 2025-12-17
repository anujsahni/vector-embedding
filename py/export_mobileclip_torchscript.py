import torch
import open_clip
from PIL import Image
from torchvision import transforms

DEVICE = "cpu"
MODEL_ID = "MobileCLIP-S1"

# Load model
model, _, _ = open_clip.create_model_and_transforms(MODEL_ID, pretrained="datacompdr", device=DEVICE)
model.eval()

# Dummy image for tracing
dummy_img = Image.new("RGB", (224, 224))
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
dummy_tensor = transform(dummy_img).unsqueeze(0).to(DEVICE)

# Trace the encode_image method
traced_model = torch.jit.trace_module(model, {"encode_image": dummy_tensor})

# Save TorchScript model
traced_model.save("mobileclip_s1_cpu.pt")
print("TorchScript model saved as mobileclip_s1_cpu.pt")
