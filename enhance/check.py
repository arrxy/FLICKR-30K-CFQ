# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: check
   Description: 二次检查enchance后的文本，删除噪音，本文件只收集promtp的回答，不进行任何处理
   Author: aidan
   date: 2023/10/1
-------------------------------------------------
"""
__author__ = 'aidan'

import json
from config import CHECK_PROMPT_LIST
import openai
from tqdm import tqdm
import random

openai.api_key = "EMPTY" # Not support yet
openai.api_base = "http://localhost:8081/v1"
# openai.api_base = "http://localhost:16363/v1"
model = "vicuna-13b-v1.1"


if __name__ == '__main__':
    text_to_enhanced_list = json.load(open('data/text_to_enhanced_list.json', 'r', encoding='utf-8'))

    text_to_raw_checked = {}
    prompt = CHECK_PROMPT_LIST[1]

    for text, enhance_list in tqdm(text_to_enhanced_list.items()):
        text_to_raw_checked[text] = {}
        for enhance_text in enhance_list:
            content = prompt.format(enhance=enhance_text, text=text)
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )
            # print the completion
            result = completion.choices[0].message.content
            text_to_raw_checked[text][enhance_text] = result
    json.dump(text_to_raw_checked, open('data/text_to_raw_checked.json', 'w', encoding='utf-8'))

