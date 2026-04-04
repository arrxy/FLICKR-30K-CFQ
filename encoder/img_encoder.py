# -*- coding: utf-8 -*-
"""
Image encoder — run one model per GPU in parallel via CUDA_VISIBLE_DEVICES.
Usage:
    CUDA_VISIBLE_DEVICES=0 python img_encoder.py --model clip-vit-base-patch32
    CUDA_VISIBLE_DEVICES=1 python img_encoder.py --model groupvit-gcc-yfcc
    CUDA_VISIBLE_DEVICES=2 python img_encoder.py --model align-base
    CUDA_VISIBLE_DEVICES=3 python img_encoder.py --model clipseg-rd64-refined
"""
import argparse
import json
import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel, AlignModel, CLIPSegModel, GroupViTModel

ALL_MODELS = ['align-base', 'clipseg-rd64-refined', 'clip-vit-base-patch32', 'groupvit-gcc-yfcc']


def load_model(model_name, model_path):
    if model_name == 'align-base':
        return AlignModel.from_pretrained(model_path)
    elif model_name == 'clipseg-rd64-refined':
        return CLIPSegModel.from_pretrained(model_path)
    elif model_name == 'clip-vit-base-patch32':
        return CLIPModel.from_pretrained(model_path)
    elif model_name == 'groupvit-gcc-yfcc':
        return GroupViTModel.from_pretrained(model_path)
    raise ValueError(f'Unknown model: {model_name}')


def encode(model_name, model_dir, test_dir, test_img_ids, device):
    model_path = os.path.join(model_dir, model_name)
    model = load_model(model_name, model_path).to(device)
    processor = AutoProcessor.from_pretrained(model_path)

    all_image_features = []
    with torch.no_grad():
        for img_id in tqdm(test_img_ids, desc=model_name):
            img_path = os.path.join(test_dir, str(img_id) + '.jpg')
            image = Image.open(img_path).convert('RGB')
            inputs = processor(images=image, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].to(device)
            feat = model.get_image_features(**inputs)
            all_image_features.append(feat.cpu())

    out_path = f'data/image_features_{model_name}.pt'
    torch.save(all_image_features, out_path)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',     default=None,
                        help='Single model name to encode. Omit to run all sequentially.')
    parser.add_argument('--model_dir', default=os.path.expanduser('~/models'),
                        help='Directory containing HuggingFace model folders')
    parser.add_argument('--test_dir',  default='../test',
                        help='Directory containing test images')
    parser.add_argument('--img_ids',   default='data/test_img_ids.json')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_img_ids = json.load(open(args.img_ids, 'r', encoding='utf-8'))
    models_to_run = [args.model] if args.model else ALL_MODELS

    for m in models_to_run:
        encode(m, args.model_dir, args.test_dir, test_img_ids, device)
