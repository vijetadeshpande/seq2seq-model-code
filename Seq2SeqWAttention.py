#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:47:48 2020

@author: vijetadeshpande
"""
import torch.nn as nn
import torch
import random

class Seq2SeqWAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, source, target, teacher_forcing_ratio = 0.5):
        
        # source = [src len, batch size]
        # target = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        # extract dimentions
        target_len = target.shape[0]
        batch_size = target.shape[1]
        target_vocab_size = self.decoder.output_dim
        
        # initialize atensor to store values of prediction at each time step
        predictions = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        enc_outputs, hidden = self.encoder(source)
        
        # first input to the decoder is the <sos> tokens
        input = target[0,:]
        
        for t in range(1, target_len):
            
            # pass the encoder outputs, hidden and input to the decoder and 
            # collect the prediction
            prediction, hidden = self.decoder(input, hidden, enc_outputs)
            
            # store prediction
            predictions[t] = prediction
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = prediction.argmax(1)
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = target[t] if teacher_force else top1
        
        
        return predictions