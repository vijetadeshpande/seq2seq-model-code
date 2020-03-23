#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 21:40:35 2020

@author: vijetadeshpande
"""
import torch.nn as nn
import torch

class DecoderWAttention(nn.Module):
    def __init__(self, output_dim, embedding_dim, dec_hidden_dim, enc_hidden_dim, 
                 dropout, attention):
        super().__init__()
        
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        
        # define unit cell
        self.rnn = nn.GRU((enc_hidden_dim * 2) + embedding_dim, dec_hidden_dim)
        
        # linear transformation 
        self.fc_out = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim + embedding_dim, output_dim)
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, dec_hidden, enc_outputs):
        
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        input = input.unsqueeze(0)
        #input = [1, batch size]
        
        # get embedded input
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, embedding_dim]
        
        # use attention layer
        a = self.attention(dec_hidden, enc_outputs)
        # a = [batch_size, seq_len]
        a = a.unsqueeze(1)
        # a = [batch_size, 1, seq_len]
        
        # reshape encoder outputs
        enc_outputs = enc_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        # use attention to weight encoder outputs
        weighted = torch.bmm(a, enc_outputs)
        # weighted = [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)        
        # weighted = [1, batch size, enc hid dim * 2]
        
        # rnn inout
        rnn_input = torch.cat((weighted, embedded), dim = 2)
        
        # rnn output
        output, hidden = self.rnn(rnn_input, dec_hidden.unsqueeze(0))
        
        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()
        
        # reshape
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        # prediction
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        # prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0)
        
        