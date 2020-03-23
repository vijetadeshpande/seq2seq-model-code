#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 00:41:11 2020

@author: vijetadeshpande
"""
import torch.nn as nn
import torch

class EncoderWAttention(nn.Module):
    def __init__(self, input_dim, embedding_dim, enc_hidden_dim, dec_hidden_dim, 
               n_layers, dropout):
        super().__init__()
        
        # embedding
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        # unit cell definition
        self.rnn = nn.GRU(input_size = embedding_dim, 
                           hidden_size = enc_hidden_dim,
                           num_layers = n_layers,
                           dropout = dropout,
                           bidirectional = True)
        
        # linear transformation on forward and backward hidden states
        self.fc = nn.Linear(enc_hidden_dim*2, dec_hidden_dim)
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, source):
        
        # dropout
        # source = [source_len, batch_size]
        embedded = self.dropout(self.embedding(source))
        # embedded = [source_len, batch_size, embedding_dim]
        
        # forward pass through unit cell of encoder
        outputs, hidden = self.rnn(embedded)
        # outputs = [source_len, batch_size, enc_hidden_dim*num_directions]
        # hidden = [n_layers*num_directions, batch_size, hidden_dim]
        
        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer
        
        # hidden [-2, :, : ] is the last of the forwards RNN 
        # hidden [-1, :, : ] is the last of the backwards RNN
        
        # initial decoder hidden is final hidden state of the forwards and backwards 
        # encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim = 1)))
        
        # hidden = [batch size, dec hid dim]
        
        return outputs, hidden
        
        
        
        
        