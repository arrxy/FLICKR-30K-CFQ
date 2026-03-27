# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: make_test_data
   Description: 
   Author: aidan
   date: 2023/9/18
-------------------------------------------------
"""
__author__ = 'aidan'

import json
from collections import defaultdict
import random

if __name__ == '__main__':
    sub_test_sen_to_ids = json.load(open('../combine/data/sub_test_sen_to_ids_300*5.json', 'r', encoding='utf-8'))

    span = int(1500 / 10 / 5)

    type_d = defaultdict(list)
    for text, item in sub_test_sen_to_ids.items():
        type_d[item['type']].append(text)

    # 10折
    test_data_11 = [[] for _ in range(10 + 1)]
    for i in range(10):
        for type, text_list in type_d.items():
            test_data_11[i].extend(text_list[i * span: (i + 1) * span])

    # 最后整体平均取150个
    all_text_list = []
    for type, text_list in type_d.items():
        all_text_list.extend(text_list)

    random.seed(20230918)
    test_data_11[10] = random.sample(all_text_list, 150)

    json.dump(test_data_11, open('data/test_data_11.json', 'w', encoding='utf-8'), ensure_ascii=False)