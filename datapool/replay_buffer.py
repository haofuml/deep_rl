#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:18:17 2018

@author: hao
"""

import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.flush()
        
    def flush(self):
        self.datapool = []
        self.pointer = 0
        
    def __len__(self):
        return len(self.datapool)
    
    def add_sample(self, ob, action, reward, done, next_ob):
        datum = (ob, action, reward, done, next_ob)
        if self.pointer == len(self.datapool):
            self.datapool.append(datum)
        else:
            self.datapool[self.pointer] = datum 
        self.pointer = (self.pointer+1)%self.max_size
        
    def next_batch(self, batch_size):
        # return (ob, action, reward, done, next_ob) where each element is a numpy array
        # with shape [batch_size, ...]
        idx = np.random.randint(0, len(self.datapool)-1, batch_size)
        batch_data = [self.datapool[i] for i in idx]
        return tuple(map(lambda x: np.array(x), zip(*batch_data)))
        
    