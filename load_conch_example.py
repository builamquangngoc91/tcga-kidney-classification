"""
Standalone CONCH model loading example.
"""
import torch
from conch.create_model import create_model_from_pretrained

# Configuration
model_cfg = 'conch_ViT-B-16'
checkpoint_path = "/project/hnguyen2/mvu9/pretrained_checkpoints/conch_checkpoints/pytorch_model.bin"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load model
model, preprocess = create_model_from_pretrained(
    model_cfg, 
    checkpoint_path, 
    device=device
)
_ = model.eval()

print("CONCH model loaded successfully!")
print(f"Device: {device}")
print(f"Model config: {model_cfg}")
