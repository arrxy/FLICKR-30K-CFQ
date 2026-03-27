# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name: tools
   Description: 
   Author: aidan
   date: 2023/8/10
-------------------------------------------------
"""
__author__ = 'aidan'

import json

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except:
            data = []
            for line in f:
                data.append(json.loads(line.strip()))
        return data