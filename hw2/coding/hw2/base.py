#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : base.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/14/2020
#
# Distributed under terms of the MIT license.

import abc


class Approach:
    """Generic approach for learning and behaving in a domain.
    """
    @abc.abstractmethod
    def set_actions(self, actions):
        """Tell the approach what actions are available in the domain

        Parameters
        ----------
        actions : [ Any ]
            For a continuous action space, this would not work! If you are
            curious how one might handle actions more generally, see
            https://gym.openai.com/docs/#spaces.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def reset(self, state):
        """Tell the approach to prepare to take actions from the given initial state.

        Parameters
        ----------
        state : pddlgym.State
            Note that the state contains the goal (state.goal).

        Returns
        -------
        info : dict
            Any logging or debugging info can go here.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def step(self, state):
        """Ask the approach for an action to take given the input state.
        Assume that the action will be subsequently executed in the environment.

        Parameters
        ----------
        state : pddlgym.State
            Note that the state contains the goal (state.goal).

        Returns
        -------
        action : Any
        info : dict
            Any logging or debugging info can go here.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def seed(self, seed):
        """Optionally set a random seed
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def train(self, env):
        """Some approaches learn. Others will do nothing for training.

        Parameters
        ----------
        env : pddlgym.PDDLEnv
            A training environment that encapsulates training problems.
        """
        raise NotImplementedError("Override me!")


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


class Featurizer:
    """Generic class for featurizers
    """
    @abc.abstractmethod
    def initialize(self, all_data):
        """Initialize the featurizer from a training dataset

        Parameters
        ----------
        all_data : [ Any ]
            A list of data.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def apply(self, x):
        """Convert a raw input to a featurized input.

        Parameters
        ----------
        x : Any
            A raw input

        Returns
        -------
        xhat : Any
            A featurized input
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def invert(self, xhat):
        """Convert a featurized input to a raw input.

        Parameters
        ----------
        x : Any
            A featurized input

        Returns
        -------
        x : Any
            A raw input
        """
        raise NotImplementedError("Override me!")


def get_approach(name, env, planning_timeout=10):
    """Put new approaches here!
    """
    if name == "random":
        from .random import RandomActions
        return RandomActions()

    if name == "astar_uniform":
        from .plan import SearchApproach, AStar
        planner = AStar(env.get_successor_state, env.check_goal, timeout=planning_timeout)
        return SearchApproach(planner=planner)

    raise Exception(f"Unrecognized approach: {name}")

