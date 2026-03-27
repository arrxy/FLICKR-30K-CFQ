# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: enhanced_text_encoder
   Description: 增强后的文本的编码器
   Author: aidan
   date: 2023/9/18
-------------------------------------------------
"""
__author__ = 'aidan'



# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: enhanceTextEncoder
   Description: 
   Author: aidan
   date: 2023/6/19
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = '../../../models'

    model_name_list = ['align-base', 'clipseg-rd64-refined', 'clip-vit-base-patch32', 'groupvit-gcc-yfcc']

    new_text_to_enhanced = json.load(open('../enhance/data/text_to_enhanced_list_2.json', 'r', encoding='utf-8'))

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

        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        all_text_features = []
        with torch.no_grad():
            for text, enhanced_text_list in new_text_to_enhanced.items():
                if len(enhanced_text_list) > 0:
                    query_list = [text + ' ' + enhanced_text for enhanced_text in enhanced_text_list]
                else:
                    query_list = [text]
                # 有一个模型不支持太长的序列
                inputs = tokenizer(query_list, padding=True, return_tensors="pt", max_length=64, truncation=True).to(device)
                text_features = model.get_text_features(**inputs)
                all_text_features.append(text_features)

        torch.save(all_text_features, 'data/enhanced_text_features_{}_2.pt'.format(model_name))
