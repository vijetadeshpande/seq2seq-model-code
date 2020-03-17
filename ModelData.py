#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:13:50 2020

@author: vijetadeshpande
"""
import pandas as pd
from torchtext import data
import torch

"""Encapsulates DataLoaders and Datasets for training, validation, test. 
Base class for fastai *Data classes."""

# create tuple of source and target
class BatchTuple():
    def __init__(self, dataset, source, target):
        self.dataset, self.source, self.target = dataset, source, target
        
    def __iter__(self):
        for batch in self.dataset:
            source = getattr(batch, self.source) 
            target = getattr(batch, self.target)                 
            yield (source, target)
            
    def __len__(self):
        return len(self.dataset)

class ModelData():
    
    def __init__(self, data_dir):
        
        # steps in getting data ready for training
        # 1. Define tokenizer
        # 2. Define torchtext.data.Field object
        # 3. Import splitted data with help of torchtext: gives you data.dataset.TabularData
        # 4. Build vocab with help of pre-trained embeddings
        # 5. Convert step3 object (which has embeddings after 4th step) to an 'iterator' object from torchtext
        # 6. Put (x, y) i.e. (source, target) in a tuple for all splits
        
        # STEP - 1:
        # torchtext lets you use the SpaCy tokenizer, which is defined in following way
        tokenizer = data.get_tokenizer('spacy')
        
        # STEP - 2:
        SRC = data.Field(tokenize = tokenizer, 
                          lower = True, 
                          init_token = '<sos>',
                          eos_token = '<eos>')
        TRG = data.Field(tokenize = tokenizer, 
                          lower = True, 
                          init_token = '<sos>',
                          eos_token = '<eos>')
        # use this object to define the data field for head and desc
        trn_data_feilds = [('source', SRC), 
                           ('target', TRG)] # source = desc, target = head
        
        # STEP - 3:
        # set path attribute and import data with torchtext
        self.path = data_dir
        trn, test = data.TabularDataset.splits(path = data_dir,
                                               train = 'trn.csv',
                                               test = 'test.csv', 
                                               format = 'csv',
                                               skip_header = True,
                                               fields = trn_data_feilds)
        
        # STEP - 4:
        # TODO: Why only training data is considered while building vocabulary?
        SRC.build_vocab(trn, vectors = 'glove.6B.100d')
        TRG.build_vocab(trn, vectors = 'glove.6B.100d')
        
        # STEP - 5:
        # Now create dataset iterater, batch, pad and numericalize
        batch_size = 8
        # USE_GPU = False
        train_iter, test_iter = data.BucketIterator.splits(
                                datasets = (trn, test), 
                                batch_sizes = (batch_size, batch_size),
                                device = 'cpu', 
                                sort_key = lambda x: len(x.source),
                                shuffle = True, sort_within_batch = False, repeat = False)
        
        # STEP - 6:
        # returns tuple of article-title pair tensors
        train_iter_tuple = BatchTuple(train_iter, "source", "target")
        test_iter_tuple = BatchTuple(test_iter, "source", "target")
        
        # set attributes for the ModelData class
        self.trn_dl, self.val_dl, self.test_dl = train_iter_tuple, None, test_iter_tuple
        self.SRC, self.TRG = SRC, TRG

    @classmethod
    def from_dls(cls, path,trn_dl,val_dl,test_dl=None):
        #trn_dl,val_dl = DataLoader(trn_dl),DataLoader(val_dl)
        #if test_dl: test_dl = DataLoader(test_dl)
        return cls(path, trn_dl, val_dl, test_dl)

    @property
    def is_reg(self): return self.trn_ds.is_reg
    @property
    def is_multi(self): return self.trn_ds.is_multi
    @property
    def trn_ds(self): return self.trn_dl.dataset
    @property
    def val_ds(self): return self.val_dl.dataset
    @property
    def test_ds(self): return self.test_dl.dataset
    @property
    def trn_y(self): return self.trn_ds.y
    @property
    def val_y(self): return self.val_ds.y