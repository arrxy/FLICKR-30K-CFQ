# -*- coding: utf-8 -*-
import sys
import json
import os
import random
from tqdm import tqdm

# ── helpers replacing the missing 'Aidan' custom library ──────────────────────
def readJson(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def writeJson(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Root of the whole project (one level up from combine/)
    main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # ── 1. Build senten_to_imgIds from combine_data ────────────────────────────
    combine_data = readJson('data/combine_data.json')
    phrase_map   = readJson('data/phrase_map.json')

    senten_to_imgIds = {}
    for img_id, info_list in tqdm(combine_data.items(), desc='building sentence map'):
        for info in info_list:
            raw_text = info['sentence']['text'].lower()
            senten_to_imgIds[raw_text] = {'type': 'rawSentence', 'images': [img_id]}

            sim_text_info = info['simSentences']
            if sim_text_info:
                sim_text = sim_text_info['text'].lower()
                senten_to_imgIds[sim_text] = {'type': 'simSentence', 'images': [img_id]}

            for fragment_info in info['fragments']:
                fragment_text = fragment_info['text'].replace('_', ' ').lower()
                if senten_to_imgIds.get(fragment_text) and img_id not in senten_to_imgIds[fragment_text]['images']:
                    senten_to_imgIds[fragment_text]['images'].append(img_id)
                elif senten_to_imgIds.get(fragment_text) is None:
                    senten_to_imgIds[fragment_text] = {'type': 'fragment', 'images': [img_id]}

            for phrase_info in info['phrases']:
                phrase_text = phrase_info['phrase'].lower()
                if senten_to_imgIds.get(phrase_text) and img_id not in senten_to_imgIds[phrase_text]['images']:
                    senten_to_imgIds[phrase_text]['images'].append(img_id)
                elif senten_to_imgIds.get(phrase_text) is None:
                    senten_to_imgIds[phrase_text] = {'type': 'phrase', 'images': [img_id]}

    # ── 2. Deduplicate similar phrases / sentences ─────────────────────────────
    sentence_map = readJson('data/sentence_map.json')

    def dedup(map_dict, threshold=0.85):
        keep, drop = [], []
        for phrase, sim_dict in tqdm(map_dict.items(), desc='dedup'):
            if phrase not in drop:
                candidates = [phrase] + [p for p, s in sim_dict.items() if s >= threshold]
                candidates.sort(key=len)
                keep.append(candidates[0])
                drop.extend(candidates[1:])
        return keep, drop

    new_phrases,   drop_phrase    = dedup(phrase_map)
    new_sentences, drop_sentences = dedup(sentence_map)

    new_senten_to_imgIds = {}
    for text, info_d in tqdm(senten_to_imgIds.items(), desc='merging'):
        type_  = info_d['type']
        img_ids = list(info_d['images'])
        if type_ == 'simSentence' and text in new_sentences:
            for sim_text, sim_score in sentence_map[text].items():
                if sim_score >= 0.85:
                    img_ids.extend(senten_to_imgIds.get(sim_text, {}).get('images', []))
        elif type_ == 'phrase' and text in new_phrases:
            for sim_text, sim_score in phrase_map[text].items():
                if sim_score >= 0.85:
                    img_ids.extend(senten_to_imgIds.get(sim_text, {}).get('images', []))
        else:
            new_senten_to_imgIds[text] = {'type': type_, 'images': img_ids}

    writeJson(new_senten_to_imgIds, 'data/senten_to_imgIds.json')

    # ── 3. Split into train / test using images in flickr30k-images/test/ ──────
    # Adjust this path to wherever your Flickr30K test images live
    test_img_dir = os.path.join(main_dir_path, 'flickr30k-images', 'test')
    if not os.path.isdir(test_img_dir):
        # Fallback: treat all images as test
        test_img_dir = os.path.join(main_dir_path, 'images')
    test_img_ids = [f[:-4] for f in os.listdir(test_img_dir) if f.endswith('.jpg')]

    sen_to_imgIds = readJson('data/senten_to_imgIds.json')

    test_sen_to_ids  = {}
    train_sen_to_ids = {}
    for text, info_d in tqdm(sen_to_imgIds.items(), desc='train/test split'):
        if all(img_id in test_img_ids for img_id in info_d['images']):
            test_sen_to_ids[text]  = info_d
        else:
            train_sen_to_ids[text] = info_d

    writeJson(train_sen_to_ids, 'data/train_sen_to_ids.json')

    # ── 4. Add image tags to test set ─────────────────────────────────────────
    img_to_tags = readJson(os.path.join(main_dir_path, 'tag', 'data', 'img_to_tags.json'))

    tag_to_imgIds = {}
    for img_id, tags in img_to_tags.items():
        for tag in tags:
            if tag not in test_sen_to_ids:
                if tag_to_imgIds.get(tag) and img_id not in tag_to_imgIds[tag]['images']:
                    tag_to_imgIds[tag]['images'].append(img_id)
                elif tag_to_imgIds.get(tag) is None:
                    tag_to_imgIds[tag] = {'type': 'tag', 'images': [img_id]}
    test_sen_to_ids.update(tag_to_imgIds)

    writeJson(test_sen_to_ids, 'data/test_sen_to_ids.json')

    # ── 5. Sample 300 per type → sub dataset ──────────────────────────────────
    type_d = {'rawSentence': [], 'simSentence': [], 'fragment': [], 'phrase': [], 'tag': []}
    for text, info_d in test_sen_to_ids.items():
        type_d[info_d['type']].append(text)

    sub_test_sen_to_ids = {}
    num = 300
    for type_, text_list in type_d.items():
        random.seed(20230618)
        random.shuffle(text_list)
        for text in text_list[:num]:
            sub_test_sen_to_ids[text] = test_sen_to_ids[text]

    writeJson(sub_test_sen_to_ids, 'data/sub_test_sen_to_ids_300*5.json')
    print(f'Done. Sub-test set: {len(sub_test_sen_to_ids)} entries.')
