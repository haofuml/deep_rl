#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:05:52 2018

@author: hao
"""

from abc import ABC, abstractmethod
import tensorflow as tf

class FunctionGraph(ABC):
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def build_graph(self, input_ph, reuse=tf.AUTO_REUSE):
        pass
    
    @abstractmethod
    def evaluate(self, input_val):
        pass
    
    def get_trainable_internal(self, scope=''):
        """ get internal trainable variables of policy object or from graph built by 
        build_graph method(build_graph need to be called under new variable scope)
        """
        # add '/' to avoid fetching variables with name as self.name
        scope += ('/'+self.name+'/') if scope else (self.name+'/')
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    
    
class Policy(FunctionGraph):
    def __init__(self, ob_ph, pd, name):
        self.ob_ph = ob_ph
        self.pd = pd
        super().__init__(name=name)
        
    def evaluate(self, obs, stochastic=False):
        feed_dict = {self.ob_ph: obs}
        if stochastic:
            return tf.get_default_session().run(self.pd.sample(), feed_dict)
        else:
            return tf.get_default_session().run(self.pd.mode(), feed_dict)
        
       
class Vfunc(FunctionGraph):
    def __init__(self, ob_ph, value, name):
        self.ob_ph = ob_ph
        self.value = value
        super().__init__(name=name)
        
    def evaluate(self, obs):
        feed_dict = {self.ob_ph: obs}
        return tf.get_default_session().run(self.value, feed_dict)
    
    
class Qfunc(FunctionGraph):
    def __init__(self, ob_ph, ac_ph, value, name):
        self.ob_ph = ob_ph
        self.ac_ph = ac_ph
        self.value = value
        super().__init__(name=name)
        
    def evaluate(self, obs, acs):
        feed_dict = {self.ob_ph: obs, self.ac_ph: acs}
        return tf.get_default_session().run(self.value, feed_dict)
    
    
    
    
    
    