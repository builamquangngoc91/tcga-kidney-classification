"""
Standalone CONCH model loading example.
"""
import torch
from conch import CONCH

# Configuration
checkpoint_path = "/project/hnguyen2/mvu9/pretrained_checkpoints/conch_checkpoints/pytorch_model.bin"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load model
model = CONCH(checkpoint_path)
model = model.to(device)
_ = model.eval()

print("CONCH model loaded successfully!")
print(f"Device: {device}")
