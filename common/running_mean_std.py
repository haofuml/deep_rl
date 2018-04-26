#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 21:09:47 2018

@author: hao
"""
import numpy as np

class RunningMeanStd(object):
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape)
        self.std = np.ones(shape)
        self.count = epsilon
        
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._parallel_update(batch_mean, batch_var, batch_count)
        
    def _parallel_update(self, batch_mean, batch_var, batch_count):
        # reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        delta = batch_mean - self.mean
        newcount = batch_count + self.count
        
        newmean = self.mean + delta * batch_count / newcount
        M = np.square(self.std) * self.count
        batchM = batch_var * batch_count
        newM = M + batchM + np.square(delta)*batch_count*self.count/newcount
        newstd = np.sqrt(newM/newcount)
        
        self.mean = newmean
        self.std = newstd
        self.count = newcount
    
class RunningMeanStd_MPI(object):
    pass
    
def test_runningmeanstd():
    for (x1, x2, x3) in [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
        (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2)),
        ]:

        rms = RunningMeanStd(epsilon=0.0, shape=x1.shape[1:])

        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [x.mean(axis=0), x.var(axis=0)]
        rms.update(x1)
        rms.update(x2)
        rms.update(x3)
        ms2 = [rms.mean, rms.var]

        assert np.allclose(ms1, ms2)
        
if __name__ == "main":
    test_runningmeanstd()