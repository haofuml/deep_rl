#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 23:22:06 2018

@author: hao
"""

import tensorflow as tf


class Pd(object):
    # a particular probability distribution with specified parameters
    def flatparam(self):
        raise NotImplementedError
        
    def sample(self):
        raise NotImplementedError
        
    def mode(self):
        raise NotImplementedError
          
    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError
        
    def logp(self, x):
        return -self.neglogp(x)
    
    def entropy(self):
        raise NotImplementedError
        
    def kl(self, other):
        raise NotImplementedError

       
class PdType(object):
    # a family of probability distribution
    def pdclass(self):
        raise NotImplementedError
        
    def pdfromflat(self, flat):
        self.pdclass()(flat)
        
    def param_shape(self):
        raise NotImplementedError
        
    def sample_shape(self):
        raise NotImplementedError
        
    def sample_dtype(self):
        raise NotImplementedError
    
class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat
    
    def pdclass(self):
        return CategoricalPd
    
    def param_shape(self):
        # (number of logits,)
        return [self.ncat]
        
    def sample_shape(self):
        return []
        
    def sample_dtype(self):
        return tf.int32

    
class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size
       
    def pdclass(self):
        return DiagGaussianPd
        
    def param_shape(self):
        # (dimension of mean + dimension of diag covariance)
        return [2*self.size]
        
    def sample_shape(self):
        return [self.size]
        
    def sample_dtype(self):
        return tf.float32
    
    
class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits
    
    def flatparam(self):
        return self.logits
        
    def sample(self):
        # using Gumbel-Max Trick and Inverse Transforming Sampling
        # https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/
        # https://en.wikipedia.org/wiki/Inverse_transform_sampling
        u = tf.random_uniform(tf.shape(self.logits))
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)
        
    def mode(self):
        return tf.argmax(self.logits, axis=-1)
          
    def neglogp(self, x):
        # cannot use tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # since it doesn't support second-order derivatives
        one_hot_x = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(logtis=self.logits, labels=one_hot_x)
               
    def entropy(self):
        a = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
        ea = tf.exp(a)
        z = tf.reduce_sum(ea, axis=-1, keep_dims=True)
        p = ea / z
        return tf.reduce_sum(p*(tf.log(z)-a), axis=-1)
        
    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keep_dims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keep_dims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0*(a0-tf.log(z0)-a1+tf.log(z1)), axis=-1)
        
class DiagGaussianPd(Pd):
    def __init__(self, mean_logstd):
        self.mean_logstd = mean_logstd
        mean, logstd = tf.split(axis=len(mean_logstd.shape)-1 ,num_or_size_splits=2, value=mean_logstd)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
       
    def flatparam(self):
        return self.mean_logstd
        
    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))
        
    def mode(self):
        return self.mean
          
    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError
        
    def entropy(self):
        raise NotImplementedError
        
    def kl(self, other):
        raise NotImplementedError
    
    