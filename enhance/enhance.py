# -*- coding: utf-8 -*-
"""
Query enhancement via local vLLM server.
Requires: vLLM server running on localhost:8081
    CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --tensor-parallel-size 2 --port 8081
"""
import json
import openai
from tqdm import tqdm
from config import PROMPT_LIST

# ── OpenAI-compatible client pointing at local vLLM ───────────────────────────
client = openai.OpenAI(
    api_key="dummy",
    base_url="http://localhost:8081/v1",
)
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"   # must match what vLLM is serving
# ──────────────────────────────────────────────────────────────────────────────

test_data = list(json.load(
    open('../combine/data/sub_test_sen_to_ids_300*5.json', 'r', encoding='utf-8')
).keys())

text_to_enhanced = {}
prompt_template = PROMPT_LIST[0]

for text in tqdm(test_data, desc='enhancing queries'):
    content = prompt_template.format(query=text)
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0.7,
        max_tokens=512,
    )
    result = completion.choices[0].message.content
    text_to_enhanced[text] = result

json.dump(text_to_enhanced,
          open('data/text_to_raw_enhanced.json', 'w', encoding='utf-8'),
          ensure_ascii=False)
print('Saved data/text_to_raw_enhanced.json')
