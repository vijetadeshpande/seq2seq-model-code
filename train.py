#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:55:06 2020

@author: vijetadeshpande
"""
import torch.nn

def train(model, iterator, optimizer, criterion, clip):
    # initiliaze
    model.train()
    epoch_loss = 0
    
    # loop over all batches in iterator
    for i, batch in enumerate(iterator):
        
        # access the source and target sequence
        src = batch.source
        trg = batch.target
        
        # make gradients equal to zero
        optimizer.zero_grad()
        
        # feed src to the encoder to get cell and hidden
        # then feed cell and hidden to deoder to get the output
        output = model(src, trg)
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        # flatten the target
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        # calculate loss
        loss = criterion(output, trg)
        
        # backward propogation
        loss.backward()
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # update weights
        optimizer.step()
        
        # update loss
        epoch_loss += loss.item()
        
    
    return epoch_loss / len(iterator)
        
    