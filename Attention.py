#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 00:15:26 2020

@author: vijetadeshpande
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        
        # attention transfomation layer
        self.attn = nn.Linear((enc_hidden_dim*2) + dec_hidden_dim, dec_hidden_dim)
        
        # learnable par
        self.v = nn.Linear(dec_hidden_dim, 1, bias = False)
        
    def forward(self, dec_hidden, enc_outputs):
        
        # dec_hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        # dim
        batch_size = dec_hidden.shape[0]
        src_len = enc_outputs.shape[0]
        
        # repeat decoder hidden state src_len times 
        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)
        enc_outputs = enc_outputs.permute(1, 0, 2)
        
        # dec_hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        # linear transformation
        energy = torch.tanh(self.attn(torch.cat((dec_hidden, enc_outputs), dim = 2)))
        #energy = [batch size, src len, dec hid dim]
        
        # attention
        attention = self.v(energy).squeeze(2)
        # attention= [batch size, src len]
        
        # standardize
        attention = F.softmax(attention, dim=1)
        
        return attention