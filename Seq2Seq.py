#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:49:32 2020

@author: vijetadeshpande
"""
import torch
import torch.nn as nn
from numpy import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        # define encoder decoder attributes
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        #
        assert encoder.hidden_dim == decoder.hidden_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
            
    def forward(self, source, target, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # define tensor to store decoder outputs ()
        batch_size = source.shape[1]
        target_len = target.shape[0]
        vocab_size = self.decoder.output_dim
        outputs = torch.zeros(target_len, batch_size, vocab_size).to(self.device)
        
        # compute cell state and hidden state from encoder forward pass
        hidden, cell = self.encoder(source)
        
        # first input to decoder is sos tokens
        input = target[0, :] # therefore, in your processes data, first element in sequence should correspond to <sos>
        
        # loop over target sequence length
        for t in range(1, target_len):
            
            # forward pass of decoder (at first 't', hidden and cell will be 
            # taken from encoder output)
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # store output
            outputs[t] = output
            
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = target[t] if teacher_force else top1
        
        return outputs
        