#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : supervised_policy.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/14/2020
#
# Distributed under terms of the MIT license.

from .base import Approach


class LearningApproach(Approach):
    def __init__(self, learner):
        self._learner = learner
        self._policy = None
        self._actions = None

    def set_actions(self, actions):
        self._actions = actions
        self._learner.set_actions(actions)

    def reset(self, obs):
        return dict()

    def step(self, obs):
        assert self._policy is not None, 'Step was called without policy learning.'
        return self._policy.step(obs)

    def train(self, env):
        self._policy = self._learner.train(env)


class MyLearningApproach1(Approach):
    """TODO: implement me!
    """
    def __init__(self):
        raise NotImplementedError("Implement me! You may want to add args or kwargs.")

    def set_actions(self, actions):
        raise NotImplementedError("Implement me!")

    def train(self, env):
        raise NotImplementedError("Implement me!")

    def reset(self, state):
        raise NotImplementedError("Implement me!")

    def step(self, obs):
        raise NotImplementedError("Implement me!")

    def seed(self, seed):
        raise NotImplementedError("Implement me!")
