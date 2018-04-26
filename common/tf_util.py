#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:20:04 2018

@author: hao
"""
import tensorflow as tf
import os
import multiprocessing

# =============================================================================
# Session Creation
# =============================================================================

def make_session(num_cpu=None, make_default=False):
    if num_cpu == None:
        num_cpu = os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count())
    
    tf_config = tf.ConfigProto(inter_op_parallelism_threads = num_cpu, 
                               intra_op_parallelism_threads = num_cpu)
    
    if make_default:
        return tf.InteractiveSession(config=tf_config)
    else:
        return tf.Session(config=tf_config)
    
def single_threaded_session(make_default=False):
    return make_session(num_cpu=1, make_default)

# =============================================================================
# Model Components
# =============================================================================

def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

# can only used when x has shape (None, dim)
def dense(x, size, name, weight_init=None, bias_init=0, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("w", [x.shape.as_list()[1], size], initializer=weight_init)
        b = tf.get_variable("b", [size], initializer=tf.constant_initializer(bias_init))
        weight_decay_fc = 3e-4
        return tf.nn.bias_add(tf.matmul(x, w), b)








