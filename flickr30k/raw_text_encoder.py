# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: text_encoder
   Description: 原文本编码器
   Author: aidan
   date: 2023/9/18
-------------------------------------------------
"""
__author__ = 'aidan'



import json
from PIL import Image
from transformers import AutoTokenizer, CLIPModel, AlignModel, CLIPSegModel, GroupViTModel
from tqdm import tqdm
import os
import torch

if __name__ == '__main__':

    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    model_dir = '../../../models'
    model_name_list = ['align-base', 'clipseg-rd64-refined', 'clip-vit-base-patch32', 'groupvit-gcc-yfcc']

    flickr30k_test_data = json.load(open('data/flickr30k_test_data.json', 'r', encoding='utf-8'))
    test_data = []
    for _, text_list in flickr30k_test_data.items():
        test_data.extend(text_list)
    print(len(test_data))
    for model_name in tqdm(model_name_list):

        model_path = os.path.join(model_dir, model_name)
        if model_name == 'align-base':
            model = AlignModel.from_pretrained(model_path)
        elif model_name == 'clipseg-rd64-refined':
            model = CLIPSegModel.from_pretrained(model_path)
        elif model_name == 'clip-vit-base-patch32':
            model = CLIPModel.from_pretrained(model_path)
        elif model_name == 'groupvit-gcc-yfcc':
            model = GroupViTModel.from_pretrained(model_path)
        # model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        with torch.no_grad():
            inputs = tokenizer(test_data, padding=True, return_tensors="pt", max_length=64, truncation=True)
            all_text_features = model.get_text_features(**inputs)

        torch.save(all_text_features, 'data/raw_text_features_{}.pt'.format(model_name))
