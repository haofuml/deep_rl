#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:00:55 2018

@author: hao
"""

import gym
from gym import spaces
import numpy as np
from common.running_mean_std import RunningMeanStd


class RmsNormalizedEnv(gym.Wrapper):
    # Todo: Reward/Return Normalization
    def __init__(self, env, normalize_ob=True, ob_clip = 10):
        gym.wrapper.__init__(env)
        self.normalize_ob = normalize_ob
        self.ob_clip = ob_clip
        self.ob_rms = RunningMeanStd(self.env.observation_space.shape[0]) if self.normalize_ob else None
        
    def step(self, act):
        ac_space = self.env.action_space
        if isinstance(ac_space, spaces.Box):
            # for continuous control, assume policy-output action has range [-1,1] approximately
            lb, ub = ac_space.low, ac_space.high
            act = np.clip(lb + (act + 1) * 0.5 * (ub - lb), lb, ub)
        ob, rew, done, info = self.env.step(act)
        if self.normalize_ob:
            self.ob_rms.update(ob)
            ob = np.clip((ob-self.ob_rms.mean)/self.ob_rms.std, -self.ob_clip, self.ob_clip)
        return ob, rew, done, info
        
        
    def reset(self):
        # Running mean and std is not reset
        ob = self.env.reset()
        if self.normalize_ob:
            self.ob_rms.update(ob)
            return np.clip((ob-self.ob_rms.mean)/self.ob_rms.std, -self.ob_clip, self.ob_clip)
        else:
            return ob
        
                
class ExpNormalizedEnv(gym.wrapper):
    pass