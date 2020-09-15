#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : supervised_policy.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/14/2020
#
# Distributed under terms of the MIT license.

from .base import Approach


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
