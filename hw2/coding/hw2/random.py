#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : random.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/14/2020
#
# Distributed under terms of the MIT license.

import numpy as np
from .base import Approach


class RandomActions(Approach):
    """Take random actions
    """
    def __init__(self):
        self._actions = None
        self._rng = None

    def set_actions(self, actions):
        self._actions = actions

    def reset(self, state):
        return {}

    def step(self, state):
        return self._rng.choice(self._actions)

    def seed(self, seed):
        self._rng = np.random.RandomState(seed)

    def train(self, env):
        pass
