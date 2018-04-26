#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 17:02:05 2018

@author: hao
"""
import tensorflow as tf


def ada_iso_gaussian_kernel(xs, ys, h_min = 1e-3):
    """
    inputs:
        xs is the fixed particle tensor with shape (None, K, dim) or (K, dim)
        ys is the updating particle tensor with shape (None, M, dim) or (M, dim)
        h_min is the minimum value of adaptive bandwidth
    outputs: 
        value for key 'output' is the kernel function k(x,y) tensor with shape (None, K, M) or (K, M)
        value for key 'gradient' is the gradient of k(x,y) with respect to x
        i.e. a tensor with shape (None, K, M, dim) or (K, M, dim)
    """
    K, dim1 = xs.shape.as_list[-2:] 
    M, dim2 = ys.shape.as_list[-2:]
    assert dim1 == dim2
    
    xs = tf.expand_dims(xs, -2) # shape (..., K, 1, dim)
    ys = tf.expand_dims(ys, -3) # shape (..., 1, M, dim)
    diff = xs-ys # shape (..., K, M, dim)
    sqr_dist = tf.reduce_sum(diff**2, axis=-1) # shape (..., K, M)
    
    num_pairs = K * M
    pre_dims = tf.shape(sqr_dist)[:-2]
    new_shape = tf.concat((pre_dims,[-1]), axis=0)
    sqr_dist_med = tf.nn.top_k(tf.reshape(sqr_dist, new_shape), num_pairs//2+1)[...,-1] # shape(...,)
    h = tf.maximum(sqr_dist_med/tf.log(K), h_min) # shape (...,)
    h = tf.stop_gradient(h) # Just in case
    
    output = tf.exp(-sqr_dist/h[..., None, None])
    gradient = (-2/h[..., None, None, None]) * output[...,None] * diff
    
    return {'output': output, 'gradient': gradient}