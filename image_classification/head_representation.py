import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import transforms
from PIL import Image
from models_vit import vit_base_patch16
from intermediate_storage import get_intermediate_x

def load_custom_model(checkpoint_path, model_arch='vit_base_patch16', num_classes=10):
    model = vit_base_patch16(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint.get("model", checkpoint)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def get_custom_last_selfattention(model, img_tensor, mask_q=None, mask_k=None, mask_attn=None, estep=(0, 11)):
    with torch.no_grad():
        model.forward(img_tensor, mask_q=mask_q, mask_k=mask_k, mask_attn=mask_attn, estep=estep)
        x = get_intermediate_x()  # Retrieve x from the utility file
        print("x.shape: ", x.shape)
        return x
    
def calculate_head_pair_frobenius_norms(x):
    print("Received x shape:", x.shape)
    num_heads = x.size(1)
    norms = {}

    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            head_diff = x[:, i, :, :] - x[:, j, :, :]
            frobenius_norm = torch.norm(head_diff, p='fro', dim=(1, 2)).mean().item()
            norms[(i, j)] = frobenius_norm

    return norms

import torchvision.transforms as transforms
from PIL import Image

def preprocess_and_save_image(image_path, resized_image_path=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))

    if resized_image_path:
        img_resized.save(resized_image_path)
    img_tensor = transform(img_resized).unsqueeze(0)

    return img_tensor, img_resized

if __name__ == '__main__':
    estep = (0, 11)
    
    fibottention_model_path = 'fibottention_model.pth'
    vit_base_model_path = 'vit_base_model.pth'
    
    fibottention_model = load_custom_model(fibottention_model_path)
    
    vit_base_model = load_custom_model(vit_base_model_path)
    image_paths = [f'img_{i}.png' for i in range(10)]

    for idx, image_path in enumerate(image_paths):
        img_tensor = preprocess_and_save_image(image_path)[0]
        fibottention_x = get_custom_last_selfattention(fibottention_model, img_tensor, estep=estep)
        vit_base_x = get_custom_last_selfattention(vit_base_model, img_tensor, estep=estep)
        fibo_pair_frobenius_norms = calculate_head_pair_frobenius_norms(fibottention_x)
        vit_pair_frobenius_norms = calculate_head_pair_frobenius_norms(vit_base_x)

        print(f"Image {idx} - Frobenius Norms for Fibottention Head Pairs:")
        for (i, j), norm in fibo_pair_frobenius_norms.items():
            print(f"  Head Pair ({i}, {j}): {norm:.4f}")

        print(f"Image {idx} - Frobenius Norms for ViT Base Head Pairs:")
        for (i, j), norm in vit_pair_frobenius_norms.items():
            print(f"  Head Pair ({i}, {j}): {norm:.4f}")
