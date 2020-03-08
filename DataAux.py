#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 22:55:37 2020

@author: vijetadeshpande
"""
import _pickle as pickle
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))

def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data

def data_to_pkl(file_path):
    # load data
    data = load_jsonl(file_path)   
    
    # separate data
    out_lib = {'head': [], 'desc': [], 'source': []}
    for i in range(0, len(data)):
        out_lib['head'].append(data[i]['title'])
        out_lib['desc'].append(data[i]['content'])
        out_lib['source'].append(data[i]['source'])
     
    # pickle the data
    with open('data.pkl', 'wb') as f:
        pickle.dump(out_lib, f)
        
    return

def save_csv(source, target, file_name):
    df = pd.DataFrame(0, index = np.arange(0, len(source)), columns = ['source', 'target'])
    df.loc[:, 'source'] = source
    df.loc[:, 'target'] = target
    
    # save csv
    df.to_csv(file_name+'.csv', header = True, index = False)
    
    return

def pkl_to_splits(pkl_path):
    # read data
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
        
    # raw data is a disctionary with keys = head, desc, source. 
    # For this project we will focus on head and desc only
    # For testing purpose, we will take a very small subset of the data
    head, desc = raw_data['head'][0:160], raw_data['desc'][0:160]
    desc_trn, desc_test, head_trn, head_test = train_test_split(desc, head)
    save_csv(desc_trn, head_trn, 'trn')
    save_csv(desc_test, head_test, 'test')
    
    return
