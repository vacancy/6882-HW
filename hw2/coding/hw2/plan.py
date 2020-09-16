#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : plan.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/14/2020
#
# Distributed under terms of the MIT license.

from collections import namedtuple, defaultdict, deque
from itertools import count, product
from tabulate import tabulate
import abc
import copy
import numpy as np
import heapq as hq
import pddlgym
from pddlgym.structs import Predicate, State, Type, LiteralConjunction
import time
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from .base import Approach


class Planner:
    """Generic class for planning
    """
    @abc.abstractmethod
    def __call__(self, state):
        """Make a plan given the state.

        Parameters
        ----------
        state : pddlgym.State
            Note that the state contains the goal (state.goal).

        Returns
        -------
        actions : [ Any ]
            The plan
        info : dict
            Any logging or debugging info can go here.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def set_actions(self, actions):
        """Tell the planner what actions are available in the domain

        Parameters
        ----------
        actions : [ Any ]
        """
        raise NotImplementedError("Override me!")


class Heuristic:
    """Generic class for heuristics
    """
    @abc.abstractmethod
    def __call__(self, node):
        """Return a heuristic value (estimated cost-to-go) given a search node.

        Parameters
        ----------
        node : AStar.Node

        Returns
        -------
        heuristic : float
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def set_actions(self, actions):
        """Tell the planner what actions are available in the domain

        Parameters
        ----------
        actions : [ Any ]
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def train(self, env):
        """Some heuristics are learnable. Others will do nothing for training.

        Parameters
        ----------
        env : pddlgym.PDDLEnv
            A training environment that encapsulates training problems.
        """
        raise NotImplementedError("Override me!")


class SearchApproach(Approach):
    """Make a plan and follow it
    """
    def __init__(self, planner, heuristic=None):
        self._planner = planner
        self._heuristic = heuristic
        self._actions = None
        self._plan = []
        self._rng = None

    def set_actions(self, actions):
        self._actions = actions
        self._planner.set_actions(actions)

    def reset(self, obs):
        self._plan, info = self._planner(obs, heuristic=self._heuristic)
        return info

    def step(self, obs):
        if not self._plan:
            print("Warning: step was called without a plan. Defaulting to random action.")
            return self._rng.choice(self._actions)
        return self._plan.pop(0)

    def seed(self, seed):
        self._rng = np.random.RandomState(seed)
        if isinstance(self._heuristic, Heuristic):
            self._heuristic.seed(seed)

    def train(self, env):
        if isinstance(self._heuristic, Heuristic):
            self._heuristic.train(env)


class AStar(Planner):
    """Planning with A* search
    """

    Node = namedtuple("Node", ["state", "parent", "action", "g"])

    def __init__(self, successor_fn, check_goal_fn, timeout=100):
        self._get_successor_state = successor_fn
        self._check_goal = check_goal_fn
        self._heuristic = None
        self._timeout = timeout
        self._actions = None

    def __call__(self, state, heuristic=None, verbose=True):
        self._heuristic = heuristic or (lambda node : 0)
        return self._get_plan(state, verbose=verbose)

    def set_actions(self, actions):
        self._actions = actions
        if isinstance(self._heuristic, Heuristic):
            self._heuristic.set_actions(actions)

    def _get_plan(self, state, verbose=True):
        start_time = time.time()
        queue = []
        state_to_best_g = defaultdict(lambda : float("inf"))
        tiebreak = count()

        root_node = self.Node(state=state, parent=None, action=None, g=0)
        hq.heappush(queue, (self._get_priority(root_node), next(tiebreak), root_node))
        num_expansions = 0

        while len(queue) > 0 and (time.time() - start_time < self._timeout):
            _, _, node = hq.heappop(queue)
            # If we already found a better path here, don't bother
            if state_to_best_g[node.state] < node.g:
                continue
            # If the goal holds, return
            if self._check_goal(node.state):
                if verbose:
                    print("\nPlan found!")
                return self._finish_plan(node), {'node_expansions' : num_expansions}
            num_expansions += 1
            if verbose:
                print(f"Expanding node {num_expansions}", end='\r', flush=True)
            # Generate successors
            for action, child_state in self._get_successors(node.state):
                # If we already found a better path to child, don't bother
                if state_to_best_g[child_state] <= node.g+1:
                    continue
                # Add new node
                child_node = self.Node(state=child_state, parent=node, action=action, g=node.g+1)
                priority = self._get_priority(child_node)
                hq.heappush(queue, (priority, next(tiebreak), child_node))
                state_to_best_g[child_state] = child_node.g

        if verbose:
            print("Warning: planning failed.")
        return [], {'node_expansions' : num_expansions}

    def _get_successors(self, state):
        for action in self._actions:
            next_state = self._get_successor_state(state, action)
            yield action, next_state

    def _finish_plan(self, node):
        plan = []
        while node.parent is not None:
            plan.append(node.action)
            node = node.parent
        plan.reverse()
        return plan

    def _get_priority(self, node):
        h = self._heuristic(node)
        if isinstance(h, tuple):
            return (tuple(node.g + hi for hi in h), h)
        return (node.g + h, h)


class BestFirstSearch(AStar):
    """Planning with best-first search
    """

    def _get_priority(self, node):
        h = self._heuristic(node)
        return h

