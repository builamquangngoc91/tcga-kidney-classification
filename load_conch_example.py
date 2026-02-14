"""
Standalone CONCH model loading and zero-shot classification example.
Based on: https://github.com/mahmoodlab/CONCH/blob/main/notebooks/zeroshot_classification_example_starter.ipynb
"""
import torch
import numpy as np
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
from PIL import Image

# Configuration
checkpoint_path = "/project/hnguyen2/mvu9/pretrained_checkpoints/conch_checkpoints/pytorch_model.bin"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load model
print("Loading CONCH model...")
model_cfg = 'conch_ViT-B-16'
model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path, device=device)
_ = model.eval()

# Get tokenizer
tokenizer = get_tokenizer()

# Define classes and prompts for kidney cancer types
classes = ['Kidney renal clear cell carcinoma (KIRC)', 
           'Kidney chromophobe carcinoma (KICH)',
           'Kidney renal papillary cell carcinoma (KIRP)']
prompts = ['an H&E image of kidney renal clear cell carcinoma',
           'an H&E image of kidney chromophobe carcinoma',
           'an H&E image of kidney renal papillary cell carcinoma']

# Example: Load an image and preprocess it
# image = Image.open('./path/to/image.jpg')
# image_tensor = preprocess(image).unsqueeze(0).to(device)

# Tokenize prompts
tokenized_prompts = tokenize(texts=prompts, tokenizer=tokenizer).to(device)
print(f"Tokenized prompts shape: {tokenized_prompts.shape}")

# Example: Encode image and text (uncomment if you have an image)
# with torch.inference_mode():
#     image_embeddings = model.encode_image(image_tensor)
#     text_embeddings = model.encode_text(tokenized_prompts)
#     sim_scores = (image_embeddings @ text_embeddings.T * model.logit_scale.exp()).softmax(dim=-1).cpu().numpy()
# 
# print("Predicted class:", classes[sim_scores.argmax()])
# print("Scores:", [f"{cls}: {score:.3f}" for cls, score in zip(classes, sim_scores[0])])

print("CONCH model and tokenizer loaded successfully!")
print(f"Device: {device}")
