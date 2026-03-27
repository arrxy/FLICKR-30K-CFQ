# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: make_test_data
   Description: 生成测试集，因为flickr30k没有做k折，所以其实只有一个测试集
   Author: aidan
   date: 2023/9/18
-------------------------------------------------
"""
__author__ = 'aidan'

import json
from collections import defaultdict
import random

if __name__ == '__main__':
    sub_test_sen_to_ids = json.load(open('data/flickr30k_test_data.json', 'r', encoding='utf-8'))



    json.dump(test_data_11, open('data/test_data_11.json', 'w', encoding='utf-8'), ensure_ascii=False)