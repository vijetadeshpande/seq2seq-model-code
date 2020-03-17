#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:49:35 2020

@author: vijetadeshpande
"""

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        
        # set attributes for the encoder
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # TODO: what if we want to use pre-trained vectors
        # define embeddings
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        # define unit cell of encoder, this is usually GRU or LSTM
        self.rnn = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, 
                           num_layers = n_layers, dropout = dropout)
        
        # define dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, source):
        
        # source = [source_len X batch_size]
        
        # embedded representation of this word will be
        # embedded_source = [source_len X batch_size X embed_dim]
        embedded = self.dropout(self.embedding(source))
        
        # take this embedding representation of the current example and pass it
        # into the LSTM
        output, (hidden, cell) = self.rnn(embedded)
        
        # dimension check:
        # out = [source_len X batch_size X hid_dim*n_directions]
        # hidden = [n_layers*n_directions, batch_size, hid_dim]
        # cell = [n_layers*n_directions, batch_size, hid_dim]
        
        return hidden, cell
    
