#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : learning.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/14/2020
#
# Distributed under terms of the MIT license.

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
        pass

class LearnedHeuristic(Heuristic):

    _state_featurizer = get_featurizer('SARMinimalState')
    _learning_model = MLPRegressor(max_iter=1000)

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


