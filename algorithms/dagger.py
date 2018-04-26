#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 11:59:11 2018

@author: hao
"""

import tensorflow as tf
from rl_algorithm import RLAlgorithm
from dataset import mujoco_dset


class Dagger(RLAlgorithm):
    def __init__(self, base_kwargs, env, policy, pool):
        super().__init__(**base_kwargs, env=env, policy=policy, pool=pool)
        
        self._creat_placeholders()
        self._build_graph()
        
    
    def _create_placeholders(self):
        pass
        
    def _get_feed_dict(self):
        pass
    
    def _build_graph(self):
        pass
       
    def _init_train(self):
        self.sess.run(tf.global_variables_initializer())
    
    
    def _do_train(self):
        pass
        
    def train(self):
        pass
        
    