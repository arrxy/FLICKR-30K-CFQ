# -*- coding: utf-8 -*-
"""
Validates each enhanced query against the original — keeps only those where
the LLM confirms visual relevance.
"""
import json
import openai
from tqdm import tqdm
from config import CHECK_PROMPT_LIST

client = openai.OpenAI(
    api_key="dummy",
    base_url="http://localhost:8081/v1",
)
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

if __name__ == '__main__':
    text_to_enhanced_list = json.load(open('data/text_to_enhanced_list.json', 'r', encoding='utf-8'))
    prompt_template = CHECK_PROMPT_LIST[1]

    text_to_raw_checked = {}
    for text, enhance_list in tqdm(text_to_enhanced_list.items(), desc='checking'):
        text_to_raw_checked[text] = {}
        for enhance_text in enhance_list:
            content = prompt_template.format(enhance=enhance_text, text=text)
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=8,
            )
            result = completion.choices[0].message.content.strip().lower()
            text_to_raw_checked[text][enhance_text] = result

    json.dump(text_to_raw_checked,
              open('data/text_to_raw_checked.json', 'w', encoding='utf-8'),
              ensure_ascii=False)
    print('Saved data/text_to_raw_checked.json')

    # Filter: keep only enhancements confirmed with "yes"
    text_to_checked_list = {}
    for text, enhance_d in text_to_raw_checked.items():
        text_to_checked_list[text] = [e for e, ans in enhance_d.items() if 'yes' in ans]

    json.dump(text_to_checked_list,
              open('data/text_to_enhanced_list_checked.json', 'w', encoding='utf-8'),
              ensure_ascii=False)
    print('Saved data/text_to_enhanced_list_checked.json')
