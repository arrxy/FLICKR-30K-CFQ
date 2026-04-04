# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: enhance
   Description: 
   Author: aidan
   date: 2023/10/1
-------------------------------------------------
"""
__author__ = 'aidan'




import openai
from tqdm import tqdm
import json
import random
from config import PROMPT_LIST



openai.api_key = "OPENAI_KEY"
openai.api_base = "https://api.openai.com/v1"
model = "gpt-3.5-turbo"

flickr30k_test_data = json.load(open('data/flickr30k_test_data.json', 'r', encoding='utf-8'))


text_to_enhanced = {}
prompt = PROMPT_LIST[0]

for _, text_list in tqdm(flickr30k_test_data.items()):
    for text in text_list:
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
        text_to_enhanced[text] = result # length=4999

json.dump(text_to_enhanced, open('data/text_to_raw_enhanced.json', 'w', encoding='utf-8'))