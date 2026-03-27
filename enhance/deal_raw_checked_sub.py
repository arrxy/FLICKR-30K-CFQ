# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: deal_raw_checked_sub
   Description: 
   Author: aidan
   date: 2023/10/13
-------------------------------------------------
"""
__author__ = 'aidan'


__author__ = 'aidan'


from collections import defaultdict
import json

if __name__ == '__main__':
    # text_to_enhanced_list = json.load(open('data/text_to_enhanced_list.json', 'r', encoding='utf-8'))
    text_to_raw_checked = json.load(open('data/sub(1,10)/text_to_raw_checked_sub.json', 'r', encoding='utf-8'))

    text_enhance_to_flag = {} # 1保留，0不要
    for text, item in text_to_raw_checked.items():
        text_enhance_to_flag[text] = {}
        for enhance_text, check_response in item.items():
            check_response = check_response.lower() # 后面出现的算
            if 'yes' in check_response:
                text_enhance_to_flag[text][enhance_text] = 1
            else:
                text_enhance_to_flag[text][enhance_text] = 0

    json.dump(text_enhance_to_flag, open('data/sub(1,10)/text_enhance_to_flag_sub.json', 'w', encoding='utf-8'))