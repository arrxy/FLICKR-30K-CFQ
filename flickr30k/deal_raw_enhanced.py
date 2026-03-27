# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: deal_raw_enhanced
   Description: 处理enhance后的文本，转化为list
   Author: aidan
   date: 2023/9/18
-------------------------------------------------
"""
__author__ = 'aidan'


import json
import re

if __name__ == '__main__':
    text_to_raw_enchanced = json.load(open('data/text_to_raw_enhanced.json', 'r', encoding='utf-8'))

    new_text_to_enhanced = {}
    for text, raw_enhanced in text_to_raw_enchanced.items():
        r = re.sub('(\d.)', ' ', raw_enhanced)
        enhance_list = r.strip().split('\n')
        new_text_to_enhanced[text] = []
        if len(enhance_list) > 1:
            for enhance in enhance_list:
                if len(enhance) > 0:
                    if enhance[0].isalpha() == False:
                        new_text_to_enhanced[text].append(enhance[1:].strip())
                    else:
                        new_text_to_enhanced[text].append(enhance.strip())
    json.dump(new_text_to_enhanced, open('data/text_to_enhanced_list.json', 'w', encoding='utf-8'), ensure_ascii=False)