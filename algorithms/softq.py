#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:34:16 2018

@author: hao
"""
from algorithms.rl_algorithm import RLAlgorithm
from common.kernel import ada_iso_gaussian_kernel
import tensorflow as tf
from functools import reduce

def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape.as_list()
    return all(a==b for a,b in zip(tensor_shape, expected_shape))



class SoftQ(RLAlgorithm):
    def __init__(self, base_kwargs, env, replay_buffer, svgd_nets, q_nets, rew_scale=1, discount=0.99,
                 qf_lr=1e-3, vf_is_size=16, target_update_interval=1000, 
                 svgd_lr=1e-3, svgd_kernel=ada_iso_gaussian_kernel , svgd_pt_size=16, update_pt_ratio=0.5):
        
        base_kwargs.update({'env':env, 'pool':replay_buffer, 'policy':svgd_nets})
        super().__init__(**base_kwargs)
        
        self.ob_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.q_func = q_nets
        self.rew_scale = rew_scale # equivalent temperature parameter
        self.discount = discount # discount factor
        
        self.qf_lr = qf_lr
        self.vf_is_size = vf_is_size # importance sampling size for evaluating value function
        self.target_update_interval = target_update_interval # update cycle for target q network
        
        self.svgd_lr = svgd_lr
        self.svgd_kernel = svgd_kernel # kernel function for svgd
        self.svgd_pt_size = svgd_pt_size # total size of particles in svgd, including training and fixed particles
        self.update_pt_ratio = update_pt_ratio
        
        self.train_ops = []
        self.target_ops = []
        
           
    def _create_placeholders(self):
        self.ob_ph = tf.placeholder(tf.float32, shape=[None, self.ob_dim], 'observations')
        self.act_ph = tf.placeholder(tf.float32, shape=[None, self.act_dim], 'actions')
        self.rew_ph = tf.placeholder(tf.float32, shape=[None], 'rewards')
        self.done_ph = tf.placeholder(tf.float32, shape=[None], 'dones')
        self.next_ob_ph = tf.placeholder(tf.float32, shape=[None, self.ob_dim], 'next_observations')
        
    def _create_q_updates(self):
        with tf.variable_scope('target'):
            next_act = tf.random_uniform([tf.shape(self.next_ob_ph)[0], self.vf_is_size, self.act_dim], -1, 1)
            q_target_vals = self.q_func.build_graph(self.next_ob_ph[:,None,:], next_act)
        assert_shape(q_target_vals, [None, self.vf_is_size])
        
        v_target_vals = tf.reduce_logsumexp(q_target_vals, axis=-1) 
        v_target_vals += (self.act_dim*tf.log(2.0) - tf.log(tf.cast(self.vf_is_size, tf.float32)))
        ys = self.rew_scale * self.rew_ph + self.discount * (1-self.done_ph) * v_target_vals
        ys = tf.stop_gradient(ys)
        assert_shape(ys, [None])
        
        q_train_vals = self.q_func.build_graph(self.ob_ph, self.act_ph, reuse=True) # set reuse=True for checking
        assert_shape(q_train_vals, [None])
        
        q_loss = 0.5 * tf.reduce_mean((ys - q_train_vals)**2)
        q_optimizer = tf.train.AdamOptimizer(self.qf_lr)
        q_update_ops = q_optimizer.minimize(loss=q_loss, var_list=self.q_func.get_trainable_internal())
        self.train_ops.append(q_update_ops)
        
        # for diagnostic
        self.q_train_vals = q_train_vals
        self.q_loss = q_loss
        
    def _create_svgd_updates(self):
        # action_pts has shape (None, K+M, act_dim)
        action_pts = self.policy.build_graph(self.ob_ph, self.svgd_pt_size, reuse=True)
        assert_shape(action_pts, [None, self.svgd_pt_size, self.act_dim]) # self.svgd_pt_size != 1 here
        
        # update_actions has shape (None, M, act_dim), fixed_actions has shape (None, K, act_dim)
        update_act_num = int(self.update_pt_ratio * self.svgd_pt_size)
        fixed_act_num = self.svgd_pt_size - update_act_num
        update_actions, fixed_actions = tf.split(action_pts, [update_act_num,fixed_act_num], axis=1)
        fixed_actions = tf.stop_gradient(fixed_actions)
        
        # kernel_output has shape (None, K, M), kernel_grad has shape (None, K, M, act_dim)
        kernel_dict = self.svgd_kernel(fixed_actions, update_actions)
        kernel_output = kernel_dict['output']
        kernel_grad = kernel_dict['gradient']
        assert_shape(kernel_output, [None, fixed_act_num, update_act_num])
        assert_shape(kernel_grad, [None, fixed_act_num, update_act_num, self.act_dim])
        
        # q_fixed_grad has shape (None, K, act_dim)
        q_fixed_vals = self.q_func.build_graph(self.ob_ph[:,None,:], fixed_actions, reuse=True)
        q_fixed_grad = tf.gradients(q_fixed_vals, fixed_actions)[0]
        q_fixed_grad = tf.stop_gradient(q_fixed_grad)
        assert_shape(q_fixed_grad, [None, fixed_act_num, self.act_dim])
        
        # opt_dir has shape (None, M, act_dim)
        opt_dir = tf.reduce_mean(kernel_output[...,None] * q_fixed_grad[:,:,None,:] + kernel_grad, axis=1)
        opt_dir = tf.stop_gradient(opt_dir)
        assert_shape(opt_dir, [None, update_act_num, self.act_dim])
        
        # svgd_grad has shape (svgd_param_size)
        svgd_params = self.policy.get_trainable_internal()
        svgd_grad = tf.gradients(update_actions, svgd_params, grad_ys=opt_dir)
        assert_shape(svgd_grad, [len(svgd_params)])
        
        # constrcut surrogate loss function with computed gradients
        # note: since svgd is gradient ascent, need to minimize -surrograte_loss
        surrogate_loss = tf.reduce_sum([tf.reduce_sum(tf.stop_gradient(grad)*param) for grad,param in zip(svgd_grad,svgd_params)])
        svgd_optimizer = tf.train.AdamOptimizer(self.svgd_lr)
        svgd_update_ops = svgd_optimizer.minimize(loss=-surrogate_loss, var_list=svgd_params)
        self.train_ops.append(svgd_update_ops)
        
                
    def _create_target_updates(self):
        # update the target Q network parameters with training Q parameters
        source_params = self.q_func.get_trainable_internal()
        target_params = self.q_func.get_trainable_internal('target')
        
        self.target_ops.extend(tf.assign(tgt, src) for tgt,src in zip(target_params,source_params))
     

    def _build_feed_dict(self, batch):
        # batch is tuple of numpy array obtained from replay buffer
        # i.e. batch = (ob, action, reward, done, next_ob)
        return {self.ob_ph: batch[0],
                self.act_ph: batch[1],
                self.rew_ph: batch[2],
                self.done_ph: batch[3],
                self.next_ob_ph: batch[4]}
    
    def _init_train(self):
        self._create_placeholders()
        self._create_q_updates()
        self._create_svgd_updates()
        self._create_target_updates()
        
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_ops)
        self.pool.flush()

    def _do_train(self, iteration, batch):
        feed_dict = self._build_feed_dict(batch)
        self.sess.run(self.train_ops, feed_dict)
        if iteration % self.target_update_interval == 0:
            # warning: now target updates can only be used when self.n_train_repeat=1
            self.sess.run(self.target_ops)
        
    
    def train(self):
        self._train()
        
    def log_diagnostics(self, batch):
        pass

    def get_snapshot(self, epoch):
        pass 
        
        
        
        
        
        