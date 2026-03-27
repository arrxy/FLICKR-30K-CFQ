# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: deal_raw_checked
   Description: 处理check过后的文本
   Author: aidan
   date: 2023/10/6
-------------------------------------------------
"""
__author__ = 'aidan'


from collections import defaultdict
import json

if __name__ == '__main__':
    text_to_enhanced_list = json.load(open('data/text_to_enhanced_list.json', 'r', encoding='utf-8'))
    text_to_raw_checked = json.load(open('data/text_to_raw_checked.json', 'r', encoding='utf-8'))

    text_enhance_to_flag = {} # 1保留，0不要
    for text, enhanced_list in text_to_enhanced_list.items():
        text_enhance_to_flag[text] = {}
        for enhance_text in enhanced_list:
            check_response = text_to_raw_checked[text][enhance_text].lower() # 后面出现的算
            if 'yes' in check_response and 'only' in check_response:
                text_enhance_to_flag[text][enhance_text] = 1
                print('a')
            else:
                text_enhance_to_flag[text][enhance_text] = 0


    json.dump(text_enhance_to_flag, open('data/text_enhance_to_flag.json', 'w', encoding='utf-8'))