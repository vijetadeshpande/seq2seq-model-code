#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:53:01 2020

@author: vijetadeshpande
"""

from ModelData import ModelData
from Encoder import Encoder
from Decoder import Decoder
from Seq2Seq import Seq2Seq
import os

# read, process and make data ready for training
all_data = ModelData(os.getcwd())
train_iterator, test_iterator = all_data.trn_ds, all_data.test_ds

#