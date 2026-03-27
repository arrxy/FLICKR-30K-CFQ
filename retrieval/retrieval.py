# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: retrieval
   Description: 
   Author: aidan
   date: 2023/9/18
-------------------------------------------------
"""
__author__ = 'aidan'

import json
import torch
from tqdm import tqdm
from collections import Counter, defaultdict

def get_text_index(text, sub_test_sen_to_ids):
    index = list(sub_test_sen_to_ids.keys()).index(text)
    return index

def retrieval_enhance(cfg):
    sub_test_sen_to_ids = json.load(open('../combine/data/sub_test_sen_to_ids_300*5.json', 'r', encoding='utf-8'))
    test_data_11 = json.load(open('data/test_data_11.json', 'r', encoding='utf-8'))
    test_img_ids = json.load(open('../encoder/data/test_img_ids.json', 'r', encoding='utf-8'))

    text_to_enhanced = json.load(open('../enhance/data/text_to_enhanced_list.json', 'r', encoding='utf-8'))

    topk_num = cfg['topk_num']

    for name in model_name_list:
        fload_recall_list = []
        print(name)
        for flod_i, test_data in enumerate(test_data_11):
            test_data_index_list = [get_text_index(text, sub_test_sen_to_ids) for text in test_data]
            image_features = torch.load('../encoder/data/image_features_{}.pt'.format(name),
                                        map_location=torch.device('cpu'))
            image_features = torch.stack(image_features).view(1000, -1)

            text_features = torch.load('../encoder/data/enhanced_text_features_{}.pt'.format(name),
                                       map_location=torch.device('cpu')) # list

            image_features /= image_features.norm(dim=-1, keepdim=True)

            recall_list = []
            for i, text in enumerate(test_data):
                enhance_list = text_to_enhanced[text]
                type = sub_test_sen_to_ids[text]['type']
                enhance_text_features = text_features[get_text_index(text, sub_test_sen_to_ids)]
                enhance_text_features /= enhance_text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * enhance_text_features @ image_features.T).softmax(dim=-1)

                true_ids = sub_test_sen_to_ids[text]['images']
                true_ids_new = [img_id for img_id in true_ids if img_id in test_img_ids]
                true_ids = true_ids_new
                counter = Counter()
                for enhance_i in range(enhance_text_features.shape[0]):
                    values, pre_indexes = similarity[enhance_i].topk(topk_num)
                    pre_ids = [test_img_ids[j] for j in pre_indexes]
                    counter.update(pre_ids)
                pre_ids = [img_id for (img_id, _) in counter.most_common(topk_num)]
                r = 0
                for true_id in true_ids:
                    if true_id in pre_ids:
                        r += 1
                recall_list.append(r / min(len(true_ids), topk_num))

            recall = sum(recall_list) / len(recall_list)
            fload_recall_list.append(recall)
            print('\tflod [{flod_i}], recall [{recall}]'.format(
                flod_i=flod_i,
                model=name,
                recall=recall
            ))
        print('model [{model}], recall [{recall}]'.format(
            model=name,
            recall=sum(fload_recall_list) / len(fload_recall_list)
        ))



def retrieval_enhance_v1(cfg):
    '''
    Counter，原query
    :param cfg:
    :return:
    '''
    sub_test_sen_to_ids = json.load(open('../combine/data/sub_test_sen_to_ids_300*5.json', 'r', encoding='utf-8'))
    test_data_11 = json.load(open('data/test_data_11.json', 'r', encoding='utf-8'))
    test_img_ids = json.load(open('../encoder/data/test_img_ids.json', 'r', encoding='utf-8'))

    text_to_enhanced = json.load(open('../enhance/data/text_to_enhanced_list.json', 'r', encoding='utf-8'))

    topk_num = cfg['topk_num']

    for name in model_name_list:
        fload_recall_list = []
        print(name)
        for flod_i, test_data in enumerate(test_data_11):
            test_data_index_list = [get_text_index(text, sub_test_sen_to_ids) for text in test_data]
            image_features = torch.load('../encoder/data/image_features_{}.pt'.format(name),
                                        map_location=torch.device('cpu'))
            image_features = torch.stack(image_features).view(1000, -1)

            text_features = torch.load('../encoder/data/enhanced_text_features_{}.pt'.format(name),
                                       map_location=torch.device('cpu')) # list

            raw_text_features = torch.load('../encoder/data/raw_text_features_{}.pt'.format(name),
                                           map_location=torch.device('cpu'))
            raw_text_features /= raw_text_features.norm(dim=-1, keepdim=True)

            image_features /= image_features.norm(dim=-1, keepdim=True)

            recall_list = []
            for i, text in enumerate(test_data):
                enhance_list = text_to_enhanced[text]
                type = sub_test_sen_to_ids[text]['type']

                index = get_text_index(text, sub_test_sen_to_ids)
                enhance_text_features = torch.cat([text_features[index], raw_text_features[index].view(1, -1)], 0)

                enhance_text_features /= enhance_text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * enhance_text_features @ image_features.T).softmax(dim=-1)

                true_ids = sub_test_sen_to_ids[text]['images']
                true_ids_new = [img_id for img_id in true_ids if img_id in test_img_ids]
                true_ids = true_ids_new
                counter = Counter()
                for enhance_i in range(enhance_text_features.shape[0]):
                    values, pre_indexes = similarity[enhance_i].topk(topk_num)
                    pre_ids = [test_img_ids[j] for j in pre_indexes]
                    counter.update(pre_ids)
                pre_ids = [img_id for (img_id, _) in counter.most_common(topk_num)]
                r = 0
                for true_id in true_ids:
                    if true_id in pre_ids:
                        r += 1
                recall_list.append(r / min(len(true_ids), topk_num))

            recall = sum(recall_list) / len(recall_list)
            fload_recall_list.append(recall)
            print('\tflod [{flod_i}], recall [{recall}]'.format(
                flod_i=flod_i,
                model=name,
                recall=recall
            ))
        print('model [{model}], recall [{recall}]'.format(
            model=name,
            recall=sum(fload_recall_list) / len(fload_recall_list)
        ))



def retrieval_enhance_v2(cfg):
    '''
    不用Counter，用相似度的平均 + 原Query
    :param cfg:
    :return:
    '''
    sub_test_sen_to_ids = json.load(open('../combine/data/sub_test_sen_to_ids_300*5.json', 'r', encoding='utf-8'))
    test_data_11 = json.load(open('data/test_data_11.json', 'r', encoding='utf-8'))
    test_img_ids = json.load(open('../encoder/data/test_img_ids.json', 'r', encoding='utf-8'))

    text_to_enhanced = json.load(open('../enhance/data/text_to_enhanced_list.json', 'r', encoding='utf-8'))

    topk_num = cfg['topk_num']

    for name in model_name_list:
        fload_recall_list = []
        print(name)
        for flod_i, test_data in enumerate(test_data_11):
            test_data_index_list = [get_text_index(text, sub_test_sen_to_ids) for text in test_data]
            image_features = torch.load('../encoder/data/image_features_{}.pt'.format(name),
                                        map_location=torch.device('cpu'))
            image_features = torch.stack(image_features).view(1000, -1)

            text_features = torch.load('../encoder/data/enhanced_text_features_{}.pt'.format(name),
                                       map_location=torch.device('cpu')) # list

            raw_text_features = torch.load('../encoder/data/raw_text_features_{}.pt'.format(name), map_location=torch.device('cpu'))
            raw_text_features /= raw_text_features.norm(dim=-1, keepdim=True)


            image_features /= image_features.norm(dim=-1, keepdim=True)

            recall_list = []
            for i, text in enumerate(test_data):
                enhance_list = text_to_enhanced[text]
                type = sub_test_sen_to_ids[text]['type']

                index = get_text_index(text, sub_test_sen_to_ids)
                enhance_text_features = torch.cat([text_features[index], raw_text_features[index].view(1, -1)], 0)

                enhance_text_features /= enhance_text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * enhance_text_features @ image_features.T).softmax(dim=-1)

                true_ids = sub_test_sen_to_ids[text]['images']
                true_ids_new = [img_id for img_id in true_ids if img_id in test_img_ids]
                true_ids = true_ids_new

                img_to_sim = defaultdict(list)
                for enhance_i in range(enhance_text_features.shape[0]):
                    values, pre_indexes = similarity[enhance_i].topk(topk_num)
                    for values, j in zip(values, pre_indexes):
                        pre_id = test_img_ids[j]
                        img_to_sim[pre_id].append(values.item())
                img_sim_sorted = sorted(img_to_sim.items(), key=lambda item: sum(item[1]) / len(item[1]), reverse=True)
                pre_ids = [pre_id for (pre_id, _) in img_sim_sorted[:topk_num]]
                r = 0
                for true_id in true_ids:
                    if true_id in pre_ids:
                        r += 1
                recall_list.append(r / min(len(true_ids), topk_num))

            recall = sum(recall_list) / len(recall_list)
            fload_recall_list.append(recall)
            print('\tflod [{flod_i}], recall [{recall}]'.format(
                flod_i=flod_i,
                model=name,
                recall=recall
            ))
        print('model [{model}], recall [{recall}]'.format(
            model=name,
            recall=sum(fload_recall_list) / len(fload_recall_list)
        ))


def retrieval_enhance_v3(cfg):
    '''

    :param cfg:
    :return:
    '''
    sub_test_sen_to_ids = json.load(open('../combine/data/sub_test_sen_to_ids_300*5.json', 'r', encoding='utf-8'))
    test_data_11 = json.load(open('data/test_data_11.json', 'r', encoding='utf-8'))
    test_img_ids = json.load(open('../encoder/data/test_img_ids.json', 'r', encoding='utf-8'))

    text_to_enhanced = json.load(open('../enhance/data/text_to_enhanced_list.json', 'r', encoding='utf-8'))

    topk_num = cfg['topk_num']

    for name in model_name_list:
        fload_recall_list = []
        print(name)
        for flod_i, test_data in enumerate(test_data_11):
            test_data_index_list = [get_text_index(text, sub_test_sen_to_ids) for text in test_data]
            image_features = torch.load('../encoder/data/image_features_{}.pt'.format(name),
                                        map_location=torch.device('cpu'))
            image_features = torch.stack(image_features).view(1000, -1)

            text_features = torch.load('../encoder/data/enhanced_text_features_{}.pt'.format(name),
                                       map_location=torch.device('cpu')) # list

            raw_text_features = torch.load('../encoder/data/raw_text_features_{}.pt'.format(name), map_location=torch.device('cpu'))
            raw_text_features /= raw_text_features.norm(dim=-1, keepdim=True)


            image_features /= image_features.norm(dim=-1, keepdim=True)

            recall_list = []
            for i, text in enumerate(test_data):
                enhance_list = text_to_enhanced[text]
                type = sub_test_sen_to_ids[text]['type']

                index = get_text_index(text, sub_test_sen_to_ids)
                enhance_text_features = torch.cat([text_features[index], raw_text_features[index].view(1, -1)], 0)

                enhance_text_features /= enhance_text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * enhance_text_features @ image_features.T).softmax(dim=-1)

                true_ids = sub_test_sen_to_ids[text]['images']
                true_ids_new = [img_id for img_id in true_ids if img_id in test_img_ids]
                true_ids = true_ids_new

                img_to_sim = defaultdict(list)

                first_reacall_img_index = []
                for enhance_i in range(enhance_text_features.shape[0]):
                    values, pre_indexes = similarity[enhance_i].topk(topk_num * 2)
                    for values, j in zip(values, pre_indexes):
                        pre_id = test_img_ids[j]
                        first_reacall_img_index.append(j.item())
                first_reacall_img_index = list(set(first_reacall_img_index))

                sub_img_features = image_features[first_reacall_img_index]
                similarity = (100.0 * enhance_text_features @ sub_img_features.T).softmax(dim=-1)

                counter = Counter()
                for enhance_i in range(enhance_text_features.shape[0]):
                    values, pre_indexes = similarity[enhance_i].topk(topk_num)
                    pre_ids = [test_img_ids[first_reacall_img_index[j]] for j in pre_indexes]
                    counter.update(pre_ids)
                pre_ids = [img_id for (img_id, _) in counter.most_common(topk_num)]

                r = 0
                for true_id in true_ids:
                    if true_id in pre_ids:
                        r += 1
                recall_list.append(r / min(len(true_ids), topk_num))

            recall = sum(recall_list) / len(recall_list)
            fload_recall_list.append(recall)
            print('\tflod [{flod_i}], recall [{recall}]'.format(
                flod_i=flod_i,
                model=name,
                recall=recall
            ))
        print('model [{model}], recall [{recall}]'.format(
            model=name,
            recall=sum(fload_recall_list) / len(fload_recall_list)
        ))

def retrieval_no_enhance(cfg):
    sub_test_sen_to_ids = json.load(open('../combine/data/sub_test_sen_to_ids_300*5.json', 'r', encoding='utf-8'))
    test_data_11 = json.load(open('data/test_data_11.json', 'r', encoding='utf-8'))
    test_img_ids = json.load(open('../encoder/data/test_img_ids.json', 'r', encoding='utf-8'))

    topk_num = cfg['topk_num']
    type_to_index = {
        'rawSentence': 0,
        'simSentence': 1,
        'fragment': 2,
        'phrase':3,
        'tag':4
    }
    for name in model_name_list:
        fload_recall_list = []
        print(name)

        type_fload_recall_list = []
        for flod_i, test_data in enumerate(test_data_11):
            test_data_index = [get_text_index(text, sub_test_sen_to_ids) for text in test_data]
            image_features = torch.load('../encoder/data/image_features_{}.pt'.format(name), map_location=torch.device('cpu'))
            image_features = torch.stack(image_features).view(1000, -1)

            text_features = torch.load('../encoder/data/raw_text_features_{}.pt'.format(name), map_location=torch.device('cpu'))
            text_features = text_features[test_data_index]
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)

            recall_list = []
            type_recall_list = [[] for _ in range(len(type_to_index))]
            for i, text in enumerate(test_data):
                info = sub_test_sen_to_ids[text]
                type = info['type']
                true_ids = info['images']


                true_ids_new = [img_id for img_id in true_ids if img_id in test_img_ids]
                true_ids = true_ids_new
                values, pre_indexes = similarity[i].topk(topk_num)

                pre_ids = [test_img_ids[j] for j in pre_indexes]
                r = 0

                for true_id in true_ids:
                    if true_id in pre_ids:
                        r += 1
                recall_list.append(r / min(len(true_ids), topk_num))

                type_recall_list[type_to_index[type]].append(r / min(len(true_ids), topk_num))
            recall = sum(recall_list) / len(recall_list)
            type_recall = [sum(r)/len(r) for r in type_recall_list]
            type_fload_recall_list.append(type_recall)
            fload_recall_list.append(recall)
            print('\tflod [{flod_i}], recall [{recall}]'.format(
                flod_i = flod_i,
                model = name,
                recall = recall,
            ))

        final_type_recall = [0 for _ in range(len(type_to_index))]
        for type_recall in type_fload_recall_list:
            for i, r in enumerate(type_recall):
                final_type_recall[i] += r
        final_type_recall = [i/len(type_fload_recall_list) for i in final_type_recall]
        print('model [{model}], recall [{recall}], type {final_type_recall}'.format(
            model=name,
            recall=sum(fload_recall_list) / len(fload_recall_list),
            final_type_recall = final_type_recall
        ))


if __name__ == '__main__':
    enhance = 'yes'

    model_name_list = ['align-base', 'clipseg-rd64-refined', 'clip-vit-base-patch32', 'groupvit-gcc-yfcc']
    cfg = {
        'topk_num': 10,
    }

    retrieval_no_enhance(cfg)

    # retrieval_enhance_v1(cfg)
    #
    # retrieval_enhance_v3(cfg)
