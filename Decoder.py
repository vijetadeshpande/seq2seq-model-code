#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 11:41:37 2020

@author: vijetadeshpande
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, embedding_dim, n_layers, dropout):
        super().__init__()
        
        #setting different dimension attributes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        # TODO: what if we want to use pre-trained embeddings
        # embedding definition
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        
        # define type of unit cell
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout)
        
        # define dropout
        self.dropout = nn.Dropout(dropout)
        
        # define tranformation on output
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input, cell, hidden):
        
        # opposed to encoder, this forward pass will be initiated with two
        # inputs cell state and hidden state
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        # unsqueeze input
        input = input.unsqueeze(0)
        # this changes input dimension to [1, batch size]
        
        # dropout
        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch size, emb dim]
        
        # forward pass with rnn
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        # transformation on output
        prediction = self.fc_out(output.squeeze(0))
        
        return prediction, hidden, cell
        
        
        