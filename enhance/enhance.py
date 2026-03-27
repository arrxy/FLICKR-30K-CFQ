# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: enhance
   Description: 对所有query进行增强
   Author: aidan
   date: 2023/9/13
-------------------------------------------------
"""
__author__ = 'aidan'



import openai
from tqdm import tqdm
import json
import random
from config import PROMPT_LIST



openai.api_key = "EMPTY" # Not support yet
openai.api_base = "http://localhost:8081/v1"
# openai.api_base = "http://localhost:16363/v1"
model = "vicuna-13b-v1.1"


test_data = list(json.load(open('../combine/data/sub_test_sen_to_ids_300*5.json', 'r', encoding='utf-8')).keys())


text_to_enhanced = {}
prompt = PROMPT_LIST[0]

for text in tqdm(test_data):
    content = prompt.format(query=text)
    # create a chat completion
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[{
         "role": "user",
         "content": content
      }]
    )
    # print the completion
    result = completion.choices[0].message.content
    text_to_enhanced[text] = result

json.dump(text_to_enhanced, open('data/text_to_raw_enhanced_3.json', 'w', encoding='utf-8'))