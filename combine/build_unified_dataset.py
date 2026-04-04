import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load sources ──────────────────────────────────────────────────────────────

# 1. Flickr30K captions: image_id → [caption0, caption1, ..., caption4]
print('Loading 30k_captions.txt ...')
raw_queries = defaultdict(list)
with open(f'{ROOT}/flickr30k/data/30k_captions.txt', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        img_tag, caption = line.split('\t', 1)
        img_id = img_tag.split('#')[0].replace('.jpg', '')
        raw_queries[img_id].append(caption)

# 2. LLaVA captions: image_id → caption string
print('Loading img_to_captions_full.json ...')
llava_captions = json.load(open(f'{ROOT}/tag/data/img_to_captions_full.json', encoding='utf-8'))

# 3. Tag responses: image_id → '#tag1 #tag2 ...'
print('Loading tag_responses_full.json ...')
tag_responses = json.load(open(f'{ROOT}/tag/data/tag_responses_full.json', encoding='utf-8'))

# 4. Test split ids (optional — for split field)
test_ids_path = f'{ROOT}/flickr30k/data/test_img_ids.json'
test_ids = set(json.load(open(test_ids_path, encoding='utf-8'))) if os.path.exists(test_ids_path) else set()


def parse_tags(raw_str):
    """'#outdoor scene #people walking' → ['outdoor scene', 'people walking']"""
    return [t.strip().lstrip('#') for t in raw_str.split('#') if t.strip()]


# ── Build dataset ─────────────────────────────────────────────────────────────
print('Building unified dataset ...')
all_img_ids = sorted(set(raw_queries) | set(llava_captions) | set(tag_responses))

dataset = []
for img_id in all_img_ids:
    dataset.append({
        "image_id": f"{img_id}.jpg",
        "queries": {
            "raw":      raw_queries.get(img_id, []),
            "similar":  [],
            "fragment": [],
            "tags":     parse_tags(tag_responses.get(img_id, '')),
            "caption":  llava_captions.get(img_id, ''),
        },
        "split": "test" if img_id in test_ids else "train",
    })

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = f'{ROOT}/Dataset/unified_dataset.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
json.dump(dataset, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

# ── Stats ─────────────────────────────────────────────────────────────────────
test_n    = sum(1 for d in dataset if d['split'] == 'test')
has_raw   = sum(1 for d in dataset if d['queries']['raw'])
has_cap   = sum(1 for d in dataset if d['queries']['caption'])
has_tags  = sum(1 for d in dataset if d['queries']['tags'])
print(f'Saved {len(dataset)} images → {out_path}')
print(f'  train: {len(dataset) - test_n}  test: {test_n}')
print(f'  with raw captions : {has_raw}')
print(f'  with llava caption: {has_cap}')
print(f'  with tags         : {has_tags}')
print(f'\nSample:')
print(json.dumps(dataset[0], indent=2))
