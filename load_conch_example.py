"""
Standalone CONCH model loading example.
"""
import torch
from conch.open_clip_custom import create_model_from_pretrained

# Configuration
checkpoint_path = "/project/hnguyen2/mvu9/pretrained_checkpoints/conch_checkpoints/pytorch_model.bin"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load model
model, preprocess = create_model_from_pretrained(
    'conch_ViT-B-16', 
    checkpoint_path, 
    device=device
)
_ = model.eval()

print("CONCH model loaded successfully!")
print(f"Device: {device}")
