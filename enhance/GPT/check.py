# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: check
   Description: 
   Author: aidan
   date: 2023/10/12
-------------------------------------------------
"""
__author__ = 'aidan'




import json
import sys
sys.path.append('/Users/aidan/Learn/NLP/Retrieval/Construction/V8')
from enhance.config import CHECK_PROMPT_LIST
from tqdm import tqdm
import random
import requests

url = ""
headers = {
    "Content-Type": "application/json"
}

if __name__ == '__main__':

    prompt = CHECK_PROMPT_LIST[1]

    text_to_enhanced_list = json.load(open('data/text_to_enhanced_list.json', 'r', encoding='utf-8'))
    test_data_11 = json.load(open('../../retrieval/data/test_data_11.json', 'r', encoding='utf-8'))

    sub_test_data = [test_data_11[2], test_data_11[7]]

    text_to_raw_checked = {}
    for test_data in sub_test_data:
        for text in tqdm(test_data):
            text_to_raw_checked[text] = {}
            enhance_list = text_to_enhanced_list[text]
            for enhance_text in enhance_list:
                content = prompt.format(enhance=enhance_text, text=text)
                data = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                    "temperature": 0.7,
                }
                post = requests.post(url, headers=headers, data=json.dumps(data))
                # print the completion
                try:
                    data = json.loads(post.text)
                except:
                    text_to_raw_checked[text][enhance_text] = 'Empty'
                else:
                    result = data["choices"][0]["message"]["content"]
                    text_to_raw_checked[text][enhance_text] = result
    json.dump(text_to_raw_checked, open('data/text_to_raw_checked.json', 'w', encoding='utf-8'))
