# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: enhance
   Description: 对所有部分query进行增强
   Author: aidan
   date: 2023/9/13
-------------------------------------------------
"""
__author__ = 'aidan'



import openai
from tqdm import tqdm
import json
import requests
import random
import sys
sys.path.append('/Users/aidan/Learn/NLP/Retrieval/Construction/V8')
from enhance.config import PROMPT_LIST


url = ""
headers = {
    "Content-Type": "application/json"
}



if __name__ == '__main__':
    text_to_enhanced = {}
    prompt = PROMPT_LIST[0]

    test_data_11 = json.load(open('../../retrieval/data/test_data_11.json', 'r', encoding='utf-8'))
    sub_test_data = [test_data_11[2], test_data_11[7]]

    for text_data in sub_test_data:
        for text in tqdm(text_data):
            content = prompt.format(query=text)
            # create a chat completion
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
                text_to_enhanced[text] = 'Empty'
            else:
                result = data["choices"][0]["message"]["content"]
                text_to_enhanced[text] = result
    json.dump(text_to_enhanced, open('data/text_to_raw_enhanced_2.json', 'w', encoding='utf-8'))