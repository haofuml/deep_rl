#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 17:20:43 2018

@author: hao
"""

import numpy as np

# =============================================================================
# Standard supversised learning dataset (Support randomization option)
# =============================================================================
class std_dset(object):
    def __init__(self, inputs, labels, random=True):
        assert len(inputs) == len(labels)
        self.inputs = inputs
        self.labels = labels
        self.size = len(self.inputs)
        self.random = random
        self._init_pointer()
    
    def _init_pointer(self):
        self.pointer = 0
        if self.random:
            idx = np.arange(self.size)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]
    
    def next_batch(self, batch_size):
        if batch_size < 0:
            return (self.inputs, self.labels)
        if self.pointer + batch_size > self.size:
            self._init_pointer()
        start = self.pointer
        self.pointer += batch_size
        return (self.inputs[start:self.pointer,:], self.labels[start:self.pointer,:])

# =============================================================================
# Mujoco supervised learning dataset 
# (For behavior clone; support trajectories with different length)        
# =============================================================================
class mujoco_dset(object):
    def __init__(self, obs, acs, rews, eps_rets, 
                 traj_limit=-1, train_frac=0.7, random=True):
        assert len(obs) == len(acs)
        assert len(acs) == len(rews)
        assert len(rews) == len(eps_rets)
        
        # keep trajectory information, stored as either 1d-array of list or 3d-array
        # (i.e. shape of eiter (e,) or (e,n,d)
        self.num_traj = len(obs) if traj_limi<0 else min(traj_limit, len(obs))
        self.obs = obs[:self.num_traj]
        self.acs = acs[:self.num_traj]
        self.rews = rews[:self.num_traj]
        self.eps_rets = eps_rets[:self.num_traj]
        
        def flatten(x):
            d = len(x[0][0])
            traj_lens = np.array([len(x_i) for x_i in x])
            flat_x = np.zeros((np.sum(traj_lens), d))
            start = 0
            for i,size in enumerate(traj_lens):
                end = start+size
                flat_x[start:end,:] = np.array(x[i])
                start = end
            return flat_x
        
        
        flat_obs = flatten(self.obs) # now has shape (num_tran, d)
        flat_acs = flatten(self.acs)
        self.num_trans = len(flat_obs)
        
        self.train_frac = train_frac
        self.random = random
        cut = int(self.num_trans * self.train_fraction)
        self.dset = std_dset(flat_obs, flat_acs, self.random)
        self.trainset = std_dset(flat_obs[:cut,:], flat_acs[:cut,:], self.random)
        self.valset = std_dset(flat_obs[cut:,:], flat_acs[cut:,:], self.random)
        
        
    
    @staticmethod
    def load(datapath, tr_frac=0.7, rand=True):
        data_npz = np.load(datapath)
        return mujoco_dset(data_npz['obs'], data_npz['acs'], data_npz['rews'], data_npz['eps_rets'],
                           train_frac=tr_frac, random=rand)
        
    def next_batch(self, dset_type=None, batch_size):
        if dset_type == None:
            self.det.next_batch(batch_size)
        elif dset_type == 'train':
            self.trainset.next_batch(batch_size)
        elif dset_type == 'dev':
            self.devset.next_batch(batch_size)
        else:
            raise NotImplementedError
            
            
            
            
            
            
            
        
    