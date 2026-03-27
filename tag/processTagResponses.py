# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: processTagResponses
   Description: 处理大模型标注完成的json文件
   Author: aidan
   date: 2023/8/10
-------------------------------------------------
"""
__author__ = 'aidan'


import json
from collections import Counter


def process(tag_file):
    img_to_tags = json.load(open(tag_file, 'r', encoding='utf-8'))
    # 文本形式的结果处理为短语list
    new_img_to_tags = {}
    for img_id, result in img_to_tags.items():
        tags = []
        word = ''
        flag = False
        for i, char in enumerate(result):
            if flag and char.isalpha() == True:
                if char.isupper() and result[i - 1].islower():
                    word += ' ' + char.lower()
                else:
                    word += char.lower()
            if char == '#':
                flag = True
                if word != '':
                    tags.append(word)
                    word = ''
            elif char.isalpha() != True:
                flag = False
        new_img_to_tags[img_id] = list(set(tags))
    return new_img_to_tags


def main(tag_file, save_file_path):
    img_to_tags = process(tag_file)

    # 统计结果
    all_tags = []
    for img_id, tags in img_to_tags.items():
        all_tags.extend(tags)
    tag_counter = Counter(all_tags)

    # 过滤
    new_all_tags = []
    for tag, num in tag_counter.items():
        if 3 <= num <= 10 and len(tag.split()) > 1:
            new_all_tags.append(tag)
    print('total [{num}] tags, like {tags}'.format(num=len(new_all_tags), tags=new_all_tags[:10]))


    new_img_to_tags = {}
    for img_id, tags in img_to_tags.items():
        new_img_to_tags[img_id] = [tag for tag in tags if tag in new_all_tags]

    json.dump(new_img_to_tags, open(save_file_path, 'w', encoding='utf-8'), ensure_ascii=False)
    print('save in {}'.format(save_file_path))

if __name__ == '__main__':
    tag_file = 'data/tag_responses.json'
    save_file_path = 'data/img_to_tags.json'

    main(tag_file, save_file_path)

