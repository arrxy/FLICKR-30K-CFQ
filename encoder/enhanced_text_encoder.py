# -*- coding: utf-8 -*-
"""
Enhanced text encoder — run one model per GPU in parallel via CUDA_VISIBLE_DEVICES.
Usage:
    CUDA_VISIBLE_DEVICES=0 python enhanced_text_encoder.py --model clip-vit-base-patch32
    CUDA_VISIBLE_DEVICES=1 python enhanced_text_encoder.py --model groupvit-gcc-yfcc
    CUDA_VISIBLE_DEVICES=2 python enhanced_text_encoder.py --model align-base
    CUDA_VISIBLE_DEVICES=3 python enhanced_text_encoder.py --model clipseg-rd64-refined
"""
import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel, AlignModel, CLIPSegModel, GroupViTModel

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


def encode(model_name, model_dir, text_to_enhanced, device):
    model_path = os.path.join(model_dir, model_name)
    model = load_model(model_name, model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    all_text_features = []
    with torch.no_grad():
        for text, enhanced_list in tqdm(text_to_enhanced.items(), desc=model_name):
            if enhanced_list:
                query_list = [text + ' ' + e for e in enhanced_list]
            else:
                query_list = [text]
            inputs = tokenizer(query_list, padding=True, return_tensors="pt",
                               max_length=64, truncation=True).to(device)
            feat = model.get_text_features(**inputs)
            all_text_features.append(feat.cpu())

    out_path = f'data/enhanced_text_features_{model_name}.pt'
    torch.save(all_text_features, out_path)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',        default=None)
    parser.add_argument('--model_dir',    default=os.path.expanduser('~/models'))
    parser.add_argument('--enhanced_file',
                        default='../enhance/data/text_to_enhanced_list.json')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_to_enhanced = json.load(open(args.enhanced_file, 'r', encoding='utf-8'))
    models_to_run = [args.model] if args.model else ALL_MODELS

    for m in models_to_run:
        encode(m, args.model_dir, text_to_enhanced, device)
