# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: img_encoder
   Description: 
   Author: aidan
   date: 2023/9/18
-------------------------------------------------
"""
__author__ = 'aidan'

import json
from PIL import Image
from transformers import AutoProcessor, CLIPModel, AlignModel, CLIPSegModel, GroupViTModel
from tqdm import tqdm
import os
import torch

if __name__ == '__main__':
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    model_dir = '../../../models'


    model_name_list = ['align-base', 'clipseg-rd64-refined', 'clip-vit-base-patch32', 'groupvit-gcc-yfcc']

    test_dir = 'img/'
    test_img_ids = json.load(open('data/test_img_ids.json', 'r', encoding='utf-8'))

    for model_name in model_name_list:

        model_path = os.path.join(model_dir, model_name)
        if model_name == 'align-base':
            model = AlignModel.from_pretrained(model_path)
        elif model_name == 'clipseg-rd64-refined':
            model = CLIPSegModel.from_pretrained(model_path)
        elif model_name == 'clip-vit-base-patch32':
            model = CLIPModel.from_pretrained(model_path)
        elif model_name == 'groupvit-gcc-yfcc':
            model = GroupViTModel.from_pretrained(model_path)
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_path)

        all_image_features = []
        with torch.no_grad():
            for img_id in tqdm(test_img_ids, desc=model_name):
                img_path = os.path.join(test_dir, str(img_id) + '.jpg')
                image = Image.open(img_path)
                inputs = processor(images=image, return_tensors="pt")
                inputs['pixel_values'] = inputs['pixel_values'].to(device)
                image_features = model.get_image_features(**inputs)
                all_image_features.append(image_features)

        torch.save(all_image_features, 'data/image_features_{}.pt'.format(model_name))


