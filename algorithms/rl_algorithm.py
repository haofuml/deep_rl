#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 00:01:24 2018

@author: hao
"""
# =============================================================================
# Base RL Algorithm class:
# =============================================================================
import abc
import tensorflow as tf
import numpy as np
from common.tf_util import make_session

class RLAlgorithm(object):
    def __init__(self, batch_size=64, on_policy=False, n_epoch=1000, epoch_length=1000, n_train_repeat=1,
                 min_pool_size=10000, max_path_length=1000, eval_n_episodes=10, eval_render=False, env, policy, pool):
        
        self.batch_size = batch_size
        self.on_policy = on_policy
        self.n_epoch = n_epoch
        self.epoch_length = epoch_length
        self.n_train_repeat = n_train_repeat
        self.min_pool_size = min_pool_size
        self.max_path_length = max_path_length
        self.eval_n_episodes = eval_n_episodes
        self.eval_render = eval_render
        
        self.sess = tf.get_default_session() or make_session(make_default=True)
        self.env = env
        self.policy = policy
        self.pool = pool
        
        
        
    @abc.abstract_method 
    def _init_train(self):
        """parameter initialization for training"""
        pass
        
    @abc.abstract_method
    def _do_train(self, iteration, batch):
        """actual training update step"""
        pass
        
    def _train(self):
        """ Perform RL training"""
        
        with self.sess.as_default():
            self._init_train()
            ob = self.env.reset()
            self.policy.reset()
                
            path_length = 0
            path_return = 0
            last_path_return = 0
            n_episodes = 0
            max_path_return = -np.inf
            
            for epoch in range(self.n_epoch):
                
                for t in range(self.epoch_length):
                    iteration = epoch * self.epoch_length + t
                    
                    action = self.policy.evaluate(ob)
                    next_ob, reward, done, info = self.env.step(action)
                    self.pool.add_sample(ob, action, reward, done, next_ob)
                    
                    path_length += 1
                    path_return += reward
                    
                    if done or self.path >= self.max_path_length:
                        ob = self.env.reset()
                        self.policy.reset()
                        n_episodes += 1
                        max_path_return = max(max_path_return, path_return)
                        last_path_return = path_return
                        path_length = 0
                        path_return = 0
                    else:
                        ob = next_ob
                    
                    if self.pool.size() >= self.min_pool_size:
                        for i in range(self.n_train_repeat):
                            batch = self.pool.next_batch(self.batch_size)
                            self._do_train(iteration, batch)
                        # for on-policy algorithm, datapool need to be empty after changing policy 
                        if self.on_policy:
                            self.pool.flush()
                self._evaluate(epoch)
                
            self.env.terminate()
            
            
    def _evaluate(self):
        raise NotImplementedError
        
    @abc.abstractmethod
    def log_diagnostics(self, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def get_snapshot(self, epoch):
        raise NotImplementedError 

    
        
        
        