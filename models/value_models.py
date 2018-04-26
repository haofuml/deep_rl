#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:46:51 2018

@author: hao
"""

from base import Vfunc, Qfunc
from gym import spaces
from common.tf_util import normc_initializer
import tensorflow as tf

"""
class usage:
    1. initialization: when creating an instance, a computational graph is built 
    under default variable scope. Specifically, placeholders and model variables
    are created, output tensors are also constructed for evaluation.
    
    2. build_graph(two basic usages; more like a function instead of class method):
        2.1 build a computational graph with new model parameters;
        need to be called under new specified variable scope.
        2.2 build a computational graph with existing model parameters;
        set reuse=True for correctness checking.
     
    3. evaluate: evaluate the output tensor(with existing model parameters)
        
    4. get_trainable_internal(two basic usages; more like a function instead of class method):
        4.1 retrieve the model parameters created with initialization
        4.2 retrieve the model parameters created with build_graph that under 
        a new variable scope       
"""


class MLP_Vfunc(Vfunc):
    def __init__(self, ob_space, hiddens, name='MLP_Vfunc'):
        assert isinstance(ob_space, spaces.Box)
        self.ob_space = ob_space
        self.hiddens = hiddens
        self.name = name
        
        ob_ph = tf.placeholder(tf.float32, [None, self.ob_space.shape[0]], 'observations')
        value = self.build_graph(ob_ph)
        super().__init__(ob_ph, value, name)
    
    
    def build_graph(self, ob_ph, reuse=tf.AUTO_REUSE):
        """
        ob_ph has shape (..., ob_dim)
        return value tensor with shape (...)
        """
        assert ob_ph.shape.as_list()[-1] == self.ob_space.shape[0]
        with tf.variable_scope(self.name, reuse=reuse):
            out = ob_ph
            activ_fn = tf.tanh
            for i,h_size in enumerate(self.hiddens):
                out = activ_fn(tf.layers.dense(out, h_size, name='fc%i'%(i+1), kernel_initializer=normc_initializer(1.0)))
            value = tf.layers.dense(out, 1, name='fcfinal', kernel_initializer=normc_initializer(1.0))
        return value[...,0]
        

class MLP_Qfunc(Qfunc):
    def __init__(self, ob_space, ac_space, hiddens, name='MLP_Qfunc'):
        assert isinstance(ob_space, spaces.Box)
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.hiddens = hiddens
        self.name = name
        
        ob_ph = tf.placeholder(tf.float32, [None,self.ob_space.shape[0]], 'observations')
        if isinstance(ac_space, spaces.Box):
            ac_ph = tf.placeholder(tf.float32, [None,self.ac_space.shape[0]], 'actions')
        elif isinstance(ac_space, spaces.Discrete):
            ac_ph = tf.placeholder(tf.int32, [None], 'actions')
        else:
            raise NotImplementedError
        value = self.build_graph(ob_ph, ac_ph)
        super().__init__(ob_ph, ac_ph, value, name)
    
    
    def build_graph(self, ob_ph, ac_ph, reuse=tf.AUTO_REUSE):
        # for ob_ph shape [..., ob_dim], ac_ph has shape [..., ac_dim] (continuous case)
        # or [...] (discrete case). Note the preceding dimension '...' need not
        # to be the same(but need to have same length) since broadcasting could be performed later.
        if isinstance(self.ac_space, spaces.Box):
            assert ac_ph.dtype == tf.float32
            assert ac_ph.shape.as_list()[-1] == self.ac_space.shape[0]
            assert len(ac_ph.shape) == len(ob_ph.shape)
            return self._build_q_cont(ob_ph, ac_ph, reuse=reuse)
        elif isinstance(self.ac_space, spaces.Discrete):
            assert ac_ph.dtype == tf.int32
            assert len(ac_ph.shape) == len(ob_ph.shape) - 1
            return self._build_q_disc(ob_ph, ac_ph, reuse=reuse)
    
 
    def _build_q_cont(self, ob_ph, ac_ph, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=reuse):
            activ_fn = tf.nn.relu
            xavier_init = tf.contrib.layers.xavier_initializer
            for i,h_size in enumerate(self.hiddens):
                if i == 0:
                    ob_out = activ_fn(tf.layers.dense(ob_ph, h_size, name='fc%iob'%(i+1), kernel_initializer=xavier_init()))
                    ac_out = activ_fn(tf.layers.dense(ac_ph, h_size, name='fc%iac'%(i+1), kernel_initializer=xavier_init()))
                    out = ob_out + ac_out
                else:
                    out = activ_fn(tf.layers.dense(out, h_size, name='fc%i'%(i+1), kernel_initializer=xavier_init()))
            value = tf.layers.dense(out, 1, name='fcfinal', kernel_initializer=xavier_init())
        return value[...,0]
    
    
    def _build_q_disc(self, ob_ph, ac_ph, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=reuse):
            activ_fn = tf.nn.relu
            xavier_init = tf.contrib.layers.xavier_initializer
            for i,h_size in enumerate(self.hiddens):
                if i == 0:
                    ob_out = activ_fn(tf.layers.dense(ob_ph, h_size, name='fc%iob'%(i+1), kernel_initializer=xavier_init()))
                    ac_out = activ_fn(tf.layers.dense(ac_ph, h_size, name='fc%iac'%(i+1), kernel_initializer=xavier_init()))
                    out = ob_out + ac_out
                else:
                    out = activ_fn(tf.layers.dense(out, h_size, name='fc%i'%(i+1), kernel_initializer=xavier_init()))
            out = tf.layers.dense(out, self.ac_space.n, name='fcfinal', kernel_initializer=xavier_init())
            value = tf.reduce_sum(out*tf.one_hot(ac_ph, self.ac_space.n), axis=-1)
        return value
        

# TODO:
class CNN_Qfunc(Qfunc):
    pass
    
    
    
    
    
    
    
    