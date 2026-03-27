# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: generate_id
   Description: 生成img的id list，方便后续对应
   Author: aidan
   date: 2023/9/18
-------------------------------------------------
"""
__author__ = 'aidan'

import os
import json

if __name__ == '__main__':
    test_dir = '/Users/aidan/Learn/NLP/Retrieval/Dataset/Flickr30k_Entities/test'
    test_img_ids = [file[:-4] for file in os.listdir(test_dir) if 'jpg' in file]

    json.dump(test_img_ids, open('data/test_img_ids.json', 'w', encoding='utf-8'), ensure_ascii=False)

