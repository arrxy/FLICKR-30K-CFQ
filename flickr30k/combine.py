# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: combine
   Description: 
   Author: aidan
   date: 2023/10/1
-------------------------------------------------
"""
__author__ = 'aidan'


import json
from collections import defaultdict



if __name__ == '__main__':

    img_file_list = []
    with open('data/flickr30k_test.txt', 'r', encoding='utf-8') as f:
        for line in f:
            img_file_list.append(line.strip())

    flickr30k_test_data = defaultdict(list)

    with open('data/30k_captions.txt', 'r', encoding='utf-8') as f:
        for line in f:
            file, text = line.strip().split('\t')
            img_file, i = file.split('#')
            if img_file in img_file_list:
                flickr30k_test_data[img_file].append(text)

    json.dump(flickr30k_test_data, open('data/flickr30k_test_data.json', 'w', encoding='utf-8'))