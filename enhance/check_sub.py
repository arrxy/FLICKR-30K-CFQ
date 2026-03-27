# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: check_sub
   Description: 只check一小部分
   Author: aidan
   date: 2023/10/12
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

    prompt = CHECK_PROMPT_LIST[1]

    text_to_enhanced_list = json.load(open('data/text_to_enhanced_list.json', 'r', encoding='utf-8'))
    test_data_11 = json.load(open('../retrieval/data/test_data_11.json', 'r', encoding='utf-8'))

    sub_test_data = [test_data_11[4], test_data_11[-1]]

    text_to_raw_checked = {}
    for test_data in sub_test_data:
        for text in tqdm(test_data):
            text_to_raw_checked[text] = {}
            enhance_list = text_to_enhanced_list[text]
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
    json.dump(text_to_raw_checked, open('data/sub(1,10)/text_to_raw_checked_sub.json', 'w', encoding='utf-8'))
