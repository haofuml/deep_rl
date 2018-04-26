#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:05:51 2018

@author: hao
"""

from base import Policy,Vfunc
from gym import spaces
from common.distributions import make_pdtype
from common.tf_util import normc_initializer
import tensorflow as tf

"""
    MLP policy network for continuous observation inputs and continuous action outputs 
    (either from sampling or deterministic)
"""
class MLP_Policy(Policy):
    def __init__(self, ob_space, ac_space, hiddens, gaussian_fixed_var=True, name='MLP_Policy'):
        assert isinstance(ob_space, spaces.Box)
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.pdtype = make_pdtype(ac_space)
        self.hiddens = hiddens
        self.gaussian_fixed_var = gaussian_fixed_var # only used for continuous action
        self.name = name
        
        ob_ph = tf.placeholder(tf.float32, [None, self.ob_space.shape[0]], 'observations')
        pd_params = self.build_graph(ob_ph)
        pd = self.pdtype.pdfromflat(pd_params)
        
        super().__init__(ob_ph, pd, name)


    def build_graph(self, ob_ph, reuse=tf.AUTO_REUSE):
        """
        ob_ph has shape (..., ob_dim)
        return action tensor with shape (...，act_dim)
        """
        assert ob_ph.shape.as_list()[-1] == self.ob_space.shape[0]
        if isinstance(self.ac_space, spaces.BOX):
            # output [mean, logstd] tensor
            return self._build_action_cont(ob_ph, reuse=reuse)
        elif isinstance(self.ac_space, spaces.Discrete):
            # output logits tensor
            return self._build_action_disc(ob_ph, reuse=reuse)
        else:
            raise NotImplementedError

  
    def _build_action_disc(self, ob_ph, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=reuse):
            out = ob_ph
            activ_fn = tf.tanh
            for i,h_size in enumerate(self.hiddens):
                out = activ_fn(tf.layers.dense(out, h_size, name='fc%i'%(i+1), kernel_initializer=normc_initializer(1.0)))
            logits = tf.layers.dense(out, self.ac_space.n, name='fcfinal', kernel_initializer=normc_initializer(0.01))
        return logits
        
        
    def _build_action_cont(self, ob_ph, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=reuse):
            out = ob_ph
            activ_fn = tf.tanh
            for i,h_size in enumerate(self.hiddens):
                out = activ_fn(tf.layers.dense(out, h_size, name='fc%i'%(i+1), kernel_initializer=normc_initializer(1.0)))
            if self.gaussian_fixed_var:
                mean = tf.layers.dense(out, self.ac_space.shape[0], name='fcfinal', kernel_initializer=normc_initializer(0.01))
                logstd = tf.get_variable('logstd', [1,self.ac_space.shape[0]], initializer=tf.zeros_initializer())
                mean_logstd = tf.concat([mean, mean*0.0+logstd], axis=-1)
            else:
                mean_logstd = tf.layers.dense(out, 2*self.ac_space.shape[0], name='fcfinal', kernel_initializer=normc_initializer(0.01))
        return mean_logstd
    
    
"""
    SVGD policy network for continuous observation inputs and continuous action particles
    to approximate implicit action distribution
    Note: subclass of Vfunc instead of Policy with self.value representing action particles
"""
class SVGD_Policy(Vfunc):
    def __init__(self, ob_space, ac_space, hiddens, squash=True, name='SVGD_Policy'):
        assert isinstance(ob_space, spaces.Box)
        assert isinstance(ac_space, spaces.Box) # SVGD can only work for continous random variable
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.hiddens = hiddens
        self.squash = squash
        self.name = name
        
        ob_ph = tf.placeholder(tf.float32, [None, self.ob_space.shape[0]], 'observations')
        value = self.build_graph(ob_ph)
        
        super().__init__(ob_ph, value, self.name)
        
        
    def build_graph(self, ob_ph, pt_size=1, reuse=tf.AUTO_REUSE):
        """
        ob_ph has shape (..., ob_dim), pt_size is the number of output action particles 
        for each single observation.
        return particle action tensor with shape (..., pt_size，act_dim) or (...，act_dim) if pt_size=1
        """
        preceding_dim = tf.shape(ob_ph)[:-1]
        if pt_size == 1:
            latent_rv = tf.random_normal(preceding_dim+[self.ac_space.shape[0]]) # shape (..., act_dim)
        else:
            ob_ph = tf.expand_dims(ob_ph, axis=-2) # shape (..., 1, ob_dim)
            latent_rv = tf.random_normal(preceding_dim+[pt_size, self.ac_space.shape[0]]) # shape (..., pt_size, act_dim)
        
        with tf.variable_scope(self.name, reuse=reuse):
            activ_fn = tf.nn.relu
            xavier_init = tf.contrib.layers.xavier_initializer
            for i,h_size in enumerate(self.hiddens):
                if i == 0:
                    ob_out = activ_fn(tf.layers.dense(ob_ph, h_size, name='fc%iob'%(i+1), kernel_initializer=xavier_init()))
                    latent_out = activ_fn(tf.layers.dense(latent_rv, h_size, name='fc%ilatent'%(i+1), kernel_initializer=xavier_init()))
                    out = ob_out + latent_out
                else:
                    out = activ_fn(tf.layers.dense(out, h_size, name='fc%i'%(i+1), kernel_initializer=xavier_init()))
            action_pts = tf.layers.dense(out, self.ac_space.n, name='fcfinal', kernel_initializer=xavier_init())
        return tf.tanh(action_pts) if self.squash else action_pts
        
          
    
class CNN_Policy(Policy):
    pass


class LSTM_Policy(Policy):
    pass





