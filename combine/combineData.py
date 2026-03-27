# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: combineData
   Description: 
   Author: aidan
   date: 2023/8/10
-------------------------------------------------
"""
__author__ = 'aidan'


import sys
sys.path.append('/Users/aidan/Learn/Tools/')
from Aidan import *
from tqdm import tqdm
import os
import random
from tqdm import tqdm
import json


if __name__ == '__main__':
    main_dir_path = '../'

    test_img_dir_path = os.path.join(main_dir_path, 'images')
    test_img_ids = [file[:-4] for file in os.listdir(test_img_dir_path) if 'jpg' in file]

    combine_data = readJson('data/combine_data.json')
    phrase_map = readJson('data/phrase_map.json')

    senten_to_imgIds = {}
    for img_id, info_list in tqdm(combine_data.items()):
        for info in info_list:
            raw_text = info['sentence']['text'].lower()
            senten_to_imgIds[raw_text] = {
                'type': 'rawSentence',
                'images': [img_id]
            }

            sim_text_info = info['simSentences']
            if sim_text_info:
                sim_text = sim_text_info['text'].lower()
                senten_to_imgIds[sim_text] = {
                    'type': 'simSentence',
                    'images': [img_id]
                }

            for fragment_info in info['fragments']:
                fragment_text = fragment_info['text'].replace('_', ' ').lower()
                if senten_to_imgIds.get(fragment_text) and img_id not in senten_to_imgIds[fragment_text]:
                    senten_to_imgIds[fragment_text]['images'].append(img_id)
                elif senten_to_imgIds.get(fragment_text) == None:
                    senten_to_imgIds[fragment_text] = {
                        'type': 'fragment',
                        'images': [img_id]
                    }

            for phrase_info in info['phrases']:
                phrase_text = phrase_info['phrase'].lower()
                if senten_to_imgIds.get(phrase_text) and img_id not in senten_to_imgIds[phrase_text]:
                    senten_to_imgIds[phrase_text]['images'].append(img_id)
                elif senten_to_imgIds.get(phrase_text) == None:
                    senten_to_imgIds[phrase_text] = {
                        'type': 'phrase',
                        'images': [img_id]
                    }

    phrase_map = readJson('data/phrase_map.json')
    sentence_map = readJson('data/sentence_map.json')

    new_phrases = []
    drop_phrase = []
    for phrase, sim_dict in tqdm(phrase_map.items()):
        if phrase not in drop_phrase:
            phrase_len = len(phrase)
            sim_phrases = [phrase]
            for sim_phrase, sim_score in sim_dict.items():
                if sim_score >= 0.85:
                    sim_phrases.append(sim_phrase)
            sim_phrases = sorted(sim_phrases, key=lambda w: len(w), reverse=False)
            new_phrases.append(sim_phrases[0])
            drop_phrase.extend(sim_phrases[1:])

    new_sentences = []
    drop_sentences = []
    for phrase, sim_dict in tqdm(sentence_map.items()):
        if phrase not in drop_sentences:
            phrase_len = len(phrase)
            sim_phrases = [phrase]
            for sim_phrase, sim_score in sim_dict.items():
                if sim_score >= 0.85:
                    sim_phrases.append(sim_phrase)
            sim_phrases = sorted(sim_phrases, key=lambda w: len(w), reverse=False)
            new_sentences.append(sim_phrases[0])
            drop_sentences.extend(sim_phrases[1:])

    new_senten_to_imgIds = {}
    for text, info_d in tqdm(senten_to_imgIds.items()):
        type = info_d['type']
        img_ids = info_d['images']
        if type == 'simSentence' and text in new_sentences:
            for sim_text, sim_score in sentence_map[text].items():
                if sim_score >= 0.85:
                    img_ids.extend(senten_to_imgIds[sim_text]['images'])
        elif type == 'phrase' and text in new_phrases:
            for sim_text, sim_score in phrase_map[text].items():
                if sim_score >= 0.85:
                    img_ids.extend(senten_to_imgIds[sim_text]['images'])
        else:
            new_senten_to_imgIds[text] = {
                'type': type,
                'images': img_ids
            }

    writeJson(new_senten_to_imgIds, 'data/senten_to_imgIds.json')

    test_dir = '/Users/aidan/Learn/NLP/Retrieval/Dataset/Flickr30k_Entities/test'
    test_img_ids = [file[:-4] for file in os.listdir(test_dir) if 'jpg' in file]

    sen_to_imgIds = json.load(open('data/senten_to_imgIds.json', 'r', encoding='utf-8'))

    test_data = []
    train_data = []
    for text, info_d in tqdm(sen_to_imgIds.items()):
        img_ids = info_d['images']
        flag = True
        for img_id in img_ids:
            if img_id not in test_img_ids:
                flag = False
                break
        if flag:
            test_data.append(text)
        else:
            train_data.append(text)

    test_sen_to_ids = {}
    train_sen_to_ids = {}
    for text, info_d in tqdm(sen_to_imgIds.items()):
        if text in test_data:
            test_sen_to_ids[text] = info_d
        else:
            train_sen_to_ids[text] = info_d

    print('save data/train_sen_to_ids.json')
    writeJson(train_sen_to_ids, 'data/train_sen_to_ids.json')

    img_to_tags = readJson('../tag/data/img_to_tags.json')

    tag_to_imgIds = {}
    for img_id, tags in img_to_tags.items():
        for tag in tags:
            if tag not in test_sen_to_ids.keys():
                if tag_to_imgIds.get(tag) and img_id not in tag_to_imgIds[tag]:
                    tag_to_imgIds[tag]['images'].append(img_id)
                elif tag_to_imgIds.get(tag) == None:
                    tag_to_imgIds[tag] = {
                        'type': 'tag',
                        'images': [img_id]
                    }
    test_sen_to_ids.update(tag_to_imgIds)

    print('save data/test_sen_to_ids.json')
    writeJson(test_sen_to_ids, 'data/test_sen_to_ids.json')

    test_sen_to_ids = readJson('data/test_sen_to_ids.json')

    type_d = {
        'rawSentence': [],
        'simSentence': [],
        'fragment': [],
        'phrase': [],
        'tag': []
    }
    for text, info_d in test_sen_to_ids.items():
        type = info_d['type']
        type_d[type].append(text)

    sub_test_sen_to_ids = {}
    num = 300
    for type, text_list in type_d.items():
        random.seed(20230618)
        random.shuffle(text_list)
        for text in text_list[:num]:
            sub_test_sen_to_ids[text] = test_sen_to_ids[text]

    writeJson(sub_test_sen_to_ids, 'data/sub_test_sen_to_ids_300*5.json')
