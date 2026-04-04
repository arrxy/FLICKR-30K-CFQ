# -*- coding: utf-8 -*-
"""Parses the numbered-list LLM output into clean JSON arrays."""
import json
import re

if __name__ == '__main__':
    text_to_raw = json.load(open('data/text_to_raw_enhanced.json', 'r', encoding='utf-8'))

    new_text_to_enhanced = {}
    for text, raw_enhanced in text_to_raw.items():
        # Strip leading "1." / "2." numbering
        r = re.sub(r'\d+\.', ' ', raw_enhanced)
        enhance_list = r.strip().split('\n')
        new_text_to_enhanced[text] = []
        if len(enhance_list) > 1:
            for enhance in enhance_list:
                enhance = enhance.strip()
                if not enhance:
                    continue
                if not enhance[0].isalpha():
                    enhance = enhance[1:].strip()
                if enhance:
                    new_text_to_enhanced[text].append(enhance)

    json.dump(new_text_to_enhanced,
              open('data/text_to_enhanced_list.json', 'w', encoding='utf-8'),
              ensure_ascii=False)
    print('Saved data/text_to_enhanced_list.json')
