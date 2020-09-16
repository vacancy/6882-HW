#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : learning.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/14/2020
#
# Distributed under terms of the MIT license.

from collections import defaultdict

import tqdm
import numpy as np
from numpy.random import RandomState
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
# from warnings import filterwarnings
# filterwarnings('ignore')

from .base import Approach, Heuristic, get_approach, run_single_test
from .featurizer import get_featurizer


class SupervisedPolicyLearning(Approach):
    def __init__(self):
        ## initialize the featurizers
        self._actions = None
        self._state_featurizer = get_featurizer('propositional')
        self._action_featurizer = get_featurizer('tabular')
        self._learning_model = DecisionTreeClassifier()

        # raise NotImplementedError("Implement me! You may want to add args or kwargs.")

    def set_actions(self, actions):
        self._actions = actions

    def train(self, env, planner_name = "astar_uniform", seed=0):
        """ train classifier """

        ## initialize the planner
        planner = get_approach(planner_name, env)
        env.reset()
        planner.set_actions(env.get_possible_actions())
        planner.seed(seed)

        ## collect data for supervised policy learning using the specified planner
        S, A = [], []
        num_data = 0
        for i in range(len(env.problems)):
            _, num_steps, _, _, states, actions = run_single_test(env, i, planner, DEBUG=False)
            S.extend(states)
            A.extend(actions)
            num_data += num_steps

        self._state_featurizer.initialize(S)
        self._action_featurizer.initialize(A)
        X = [self._state_featurizer.apply(s) for s in S]
        Y = [self._action_featurizer.apply(a) for a in A]
        self._learning_model.fit(X,Y)
        # print(f'Fit DecisionTreeClassifier on {num_data} (S, A) pairs\n')

    def reset(self, state):
        return {'node_expansions': 0}

    def step(self, obs):
        """ test classifier """
        X = self._state_featurizer.apply(obs).reshape(1, -1)
        y = self._action_featurizer.invert(self._learning_model.predict(X)[0])
        return y

    def seed(self, seed):
        pass


class SupervisedHeuristicLearning(Approach):
    def __init__(self, planner, heuristic):
        ## initialize the featurizers
        self._actions = None
        self._planner = planner
        self._heuristic = heuristic
        self._plan = []

    def set_actions(self, actions):
        self._actions = actions
        self._planner.set_actions(actions)
        if isinstance(self._heuristic, Heuristic):
            self._heuristic.set_actions(actions)

    def train(self, env):
        if isinstance(self._heuristic, Heuristic):
            self._heuristic.train(env)

    def reset(self, obs):
        self._plan, info = self._planner(obs, heuristic=self._heuristic, verbose=False)
        return info

    def step(self, obs):
        return self._plan.pop(0)

    def seed(self, seed):
        # NB(Jiayuan Mao @ 09/16): for QLearning heuristic, we need to set the random seed (for eps-greedy).
        if hasattr(self._heuristic, 'seed'):
            self._heuristic.seed(seed)


class LearnedHeuristic(Heuristic):
    def __init__(self):
        self._state_featurizer = get_featurizer('SARMinimalState')
        self._learning_model = MLPRegressor(max_iter=1000)

    def __call__(self, node):
        X = self._state_featurizer.apply(node.state).reshape(1, -1)
        hs = self._learning_model.predict(X)[0]
        return hs

    def set_actions(self, actions):
        self._actions = actions

    def train(self, env):
        ## initialize the planner
        planner = get_approach("astar_uniform", env)
        env.reset()
        planner.set_actions(env.get_possible_actions())
        planner.seed(0)

        ## collect data for supervised heuristic learning using the specified planner
        S, Y = [], []
        num_data = 0
        for i in range(len(env.problems)):
            _, num_steps, _, _, states, _ = run_single_test(env, i, planner, DEBUG=False)
            S.extend(states)
            hs = list(range(len(states)))
            hs.reverse()
            Y.extend(hs)
            num_data += num_steps

        self._state_featurizer.initialize(S)
        X = [self._state_featurizer.apply(s) for s in S]
        self._learning_model.fit(X, Y)
        # print(f'Fit MLPRegressor on {num_data} (S, hs) pairs\n')



class TabularQ(object):
    def __init__(self, actions, gamma):
        self._actions = actions
        self._gamma = gamma
        self.q = defaultdict(dict)

    def q(self, state, action, default=0):
        if state in self.q:
            return self.q[state].get(action, default=default)
        return default

    def action(self, state, rng, epsilon=None):
        if state not in self.q or (epsilon is not None and rng.rand() < epsilon):
            return rng.choice(self._actions)

        values = self.q[state]
        return max(values.keys(), key=values.get)

    def value(self, state):
        if state not in self.q:
            return 0
        values = self.q[state]
        return max(values.values())

    def update(self, state, action, next_state, r):
        this_q = r + self._gamma * self.value(next_state)
        self.q[state][action] = this_q


class QLearningHeuristic(Heuristic):
    def __init__(self, H, T, gamma, epsilon):
        self._H = H
        self._T = T
        self._gamma = gamma
        self._epsilon = epsilon
        self._rng = RandomState()

        self._state_featurizer = get_featurizer('SARMinimalState')
        self._action_featurizer = get_featurizer('tabular', one_hot=True)
        self._learning_model = MLPRegressor(max_iter=1000, learning_rate_init=0.5)

    def __call__(self, node):
        X = self._state_featurizer.apply(node.state).reshape(1, -1)

        best_q = -1e9
        for action in self._actions:
            A = self._action_featurizer.apply(action).reshape(1, -1)
            q = self._learning_model.predict(np.concatenate([X, A], axis=-1))
            best_q = max(best_q, q)

        return best_q

    def set_actions(self, actions):
        self._actions = actions

    def train(self, env):
        tabular_q = TabularQ(self._actions, self._gamma)

        state, done = None, True
        step = 0
        for i in tqdm.tqdm(range(self._T), desc='QLearning'):
            if done or step >= self._H:
                state, info = env.reset()
            action = tabular_q.action(state, self._rng, self._epsilon)
            next_state, reward, done, info = env.step(action)
            tabular_q.update(state, action, next_state, reward)
            state = next_state

        S, A, Y = list(), list(), list()
        for state, action_qvalue_pairs in tabular_q.q.items():
            for action, q_value in action_qvalue_pairs.items():
                S.append(state)
                A.append(action)
                Y.append(q_value)

        self._state_featurizer.initialize(S)
        self._action_featurizer.initialize(A)

        X = list()
        for state, actoin in zip(S, A):
            X.append(np.concatenate(
                [self._state_featurizer.apply(state), self._action_featurizer.apply(action)],
                axis=-1
            ))
        X = np.stack(X, axis=0)
        Y = np.stack(Y, axis=0)
        self._learning_model.fit(X, Y)

    def seed(self, seed):
        self._rng.seed(seed)
