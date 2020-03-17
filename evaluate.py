#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 20:47:35 2020

@author: vijetadeshpande
"""
import torch

def evaluate(model, iterator, criterion):
    
    # initialize
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            
            # read source and target values
            src = batch.source
            trg = batch.target
            
            # predict output
            output = model(src, trg, 0) # switch off teacher forcing
            # dimension check:
            # trg = [target_len, batch_size]
            # output = [target_len, batch_size X output_dim]
            
            # calculate error
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            # dimension check
            # output = [(target_len - 1)*batch_size, output_dim]
            # trg = [(target_len - 1)*batch_size]
            loss = criterion(output, trg)
            
            # update error
            epoch_loss += loss.item()
            
    return epoch_loss/len(iterator)
            