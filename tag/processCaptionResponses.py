# -*- coding: utf-8 -*-
"""
Cleans raw LLaVA caption output into a single clean string per image.
Input:  data/caption_responses_full.json  →  {img_id: raw_caption_text}
Output: data/img_to_captions_full.json   →  {img_id: clean_caption}
"""
import json
import re


def clean_caption(raw: str) -> str:
    """Keep only the first 1-2 sentences, strip leading/trailing whitespace."""
    raw = raw.strip()
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', raw)
    # Take up to 2 sentences
    caption = ' '.join(sentences[:2]).strip()
    return caption


if __name__ == '__main__':
    caption_file = 'data/caption_responses_full.json'
    save_file_path = 'data/img_to_captions_full.json'

    raw_captions = json.load(open(caption_file, 'r', encoding='utf-8'))

    img_to_captions = {}
    for img_id, raw in raw_captions.items():
        img_to_captions[img_id] = clean_caption(raw)

    json.dump(img_to_captions, open(save_file_path, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print(f'Saved {len(img_to_captions)} captions to {save_file_path}')
    # Sample
    for img_id, cap in list(img_to_captions.items())[:3]:
        print(f'  {img_id}: {cap}')
