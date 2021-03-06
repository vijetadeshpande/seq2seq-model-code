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
from EncoderWAttention import EncoderWAttention
from DecoderWAttention import DecoderWAttention
from Seq2SeqWAttention import Seq2SeqWAttention
from Attention import Attention
from train import train
from evaluate import evaluate
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
import math

# read, process and make data ready for training
all_data = ModelData(os.getcwd())
train_iterator, test_iterator = all_data.trn_ds, all_data.test_ds

# extract Field object from the DataModel object
SRC, TRG = all_data.SRC, all_data.TRG

# intialization hyper-par
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 100
DEC_EMB_DIM = 100
ENC_HID_DIM = 2
DEC_HID_DIM = 2
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize encoder, decoder and model onject
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

# initialize encoder, decoder, model objects with attention
attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc_a = EncoderWAttention(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, N_LAYERS, ENC_DROPOUT)
dec_a = DecoderWAttention(OUTPUT_DIM, DEC_EMB_DIM, DEC_HID_DIM, ENC_HID_DIM, DEC_DROPOUT, attn)
model_a = Seq2SeqWAttention(enc_a, dec_a, device).to(device)

# initialize weights
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
model.apply(init_weights)
model_a.apply(init_weights)

# count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model w/o attention has {count_parameters(model):,} trainable parameters')
print(f'The model with attention has {count_parameters(model_a):,} trainable parameters')


# define optimizer
optimizer = optim.Adam(model.parameters())

# define error function (ignore padding and sos/eos tokens)
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

# training parameters
N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')

# auxilliary function
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# start training without attention
for epoch in range(N_EPOCHS):
    
    # WITHOUT ATTENTION
    # start clock
    start_time = time.time()
    # train
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    # stop clock
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    # print
    # if we happen to have a validation data set then calculate validation
    # loss here by predicting the value of validation set 'x's
    valid_loss = 0
    # update validation loss if less than previously observed minimum
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')        

    print('WITHOUT ATTENTION: ')
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
    # WITH ATTENTION
    # start clock
    start_time = time.time()
    # train
    train_loss_a = train(model_a, train_iterator, optimizer, criterion, CLIP)
    # stop clock
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    # print
    # if we happen to have a validation data set then calculate validation
    # loss here by predicting the value of validation set 'x's
    valid_loss = 0
    # update validation loss if less than previously observed minimum
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print('WITH ATTENTION: ')
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss_a:.3f} | Train PPL: {math.exp(train_loss_a):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    print('\n')
    

# testing/prediction
;model.load_state_dict(torch.load('tut1-model.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')



    

    
    

