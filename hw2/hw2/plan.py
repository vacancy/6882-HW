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
from tqdm import tqdm
import abc
import numpy as np
import heapq as hq
import time

from .base import Approach
from .learning import TabularQ


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

    def __call__(self, state, heuristic=None, verbose=False):
        self._heuristic = heuristic or (lambda node : 0)
        return self._get_plan(state, verbose=verbose)

    def set_actions(self, actions):
        self._actions = actions
        if isinstance(self._heuristic, Heuristic):
            self._heuristic.set_actions(actions)

    def _get_plan(self, state, verbose=False):
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


class UCT(Planner):
    """Implementation of UCT based on Leslie's lecture notes

    """
    def __init__(self, successor_fn, check_goal_fn, reward_fn,
                 num_search_iters=100, timeout=100,
                 replanning_interval=5, max_num_steps=250, gamma=0.9, seed=0):

        self._get_successor_state = successor_fn
        self._check_goal = check_goal_fn
        self._reward_fn = reward_fn

        self._num_search_iters = num_search_iters
        self._max_num_steps = max_num_steps
        self._gamma = gamma
        self._timeout = timeout
        self._replanning_interval = replanning_interval

        self._rng = np.random.RandomState(seed)
        self._actions = None
        self._Q = None
        self._N = None

    def __call__(self, state, heuristic=None, verbose=False):
        steps_since_replanning = 0
        plan = []
        for t in tqdm(range(self._max_num_steps)):
            if t % self._replanning_interval == 0:
                if verbose: print("Running UCT...")
                self.run(state, horizon=self._max_num_steps - t)
                steps_since_replanning = 0
                if verbose: print("Done.")

            # print(f'generating action at step: {t}')
            # values = {a: self._Q[state][a][t] for a in self._actions}
            # print(sorted(values.items(), key=lambda x: -x[1]))

            action = self._get_action(state, t=steps_since_replanning)
            steps_since_replanning += 1
            # if verbose: print("Taking action", action)
            state = self._get_successor_state(state, action)
            # if verbose: print("Next state:", state)
            plan.append(action)
            if self._check_goal(state):
                if verbose: print("!! UCT found goal", action)
                break

        return plan, {'node_expansions': 0}

    def run(self, state, horizon=100):
        # Initialize Q[s][a][d] -> float
        self._Q = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
        # Initialize N[s][a][d] -> int
        self._N = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
        # Loop search
        start_time = time.time()
        for it in tqdm(range(self._num_search_iters), desc='UCT Search'):
            if time.time() - start_time > self._timeout:
                print(f'.... stopped UCT after {it} iterations (timeout = {self._timeout})')
                break
            # Update Q
            self._search(state, 0, horizon=horizon)

    def set_actions(self, actions):
        self._actions = actions

    def _get_action(self, state, t=0):
        # Return best action, break ties randomly
        return max(self._actions, key=lambda a : (self._Q[state][a][t], self._rng.uniform()))

    def _search(self, s, depth, horizon):
        # Base case
        if depth == horizon:
            return 0.

        # Select an action, balancing explore/exploit
        a = self._select_action(s, depth, horizon=horizon)
        # Create a child state
        next_state = self._get_successor_state(s, a)

        # Get value estimate
        if self._check_goal(next_state):
            # Some environments terminate problems before the horizon
            reward = self._reward_fn(s, a)
            reward = 100
            q = reward
        else:
            reward = self._reward_fn(s, a)
            reward = -1
            q = reward + self._gamma * self._search(next_state, depth+1, horizon=horizon)
        # Update values and counts
        num_visits = self._N[s][a][depth] # before now

        # First visit to (s, a, depth)
        if num_visits == 0:
            self._Q[s][a][depth] = q
        # We've been here before
        else:
            # Running average
            self._Q[s][a][depth] = (num_visits / (num_visits + 1.)) * self._Q[s][a][depth] + \
                                   (1 / (num_visits + 1.)) * q
        # Update num visits
        self._N[s][a][depth] += 1
        return self._Q[s][a][depth]

    def _select_action(self, s, depth, horizon):
        # If there is any action where N(s, a, depth) == 0, try it first
        untried_actions = [a for a in self._actions if self._N[s][a][depth] == 0]
        if len(untried_actions) > 0:
            return self._rng.choice(untried_actions)
        # Otherwise, take an action to trade off exploration and exploitation
        N_s_d = sum(self._N[s][a][depth] for a in self._actions)
        best_action_score = -np.inf
        best_actions = []
        for a in self._actions:
            explore_bonus = (np.log(N_s_d) / self._N[s][a][depth])**((horizon + depth) / (2*horizon + depth))
            # explore_bonus = (np.log(N_s_d) / self._N[s][a][depth])**(0.5)
            # explore_bonus = 0
            score = self._Q[s][a][depth] + explore_bonus
            if score > best_action_score:
                best_action_score = score
                best_actions = [a]
            elif score == best_action_score:
                best_actions.append(a)
        return self._rng.choice(best_actions)


class DPApproach(Approach):
    def __init__(self, planner):
        self._planner = planner
        self._actions = None
        self._plan = []

    def set_actions(self, actions):
        self._actions = actions

    def reset(self, obs):
        self._plan, info = self._planner(obs)
        return info

    def step(self, obs):
        if not self._plan:
            print("Warning: step was called without a plan. Defaulting to random action.")
            return self._rng.choice(self._actions)
        return self._plan.pop(0)

    def seed(self, seed):
        self._rng = np.random.RandomState(seed)

    def train(self, env):
        pass


class VI(DPApproach):
    def __init__(self, states, successor_fn, check_goal_fn, reward_fn,
                 gamma=0.9, epsilon=0.1, timeout=100, seed=0):

        self._states = None
        self._actions = None
        self.T = successor_fn
        self.R = reward_fn
        self._check_goal = check_goal_fn

        self._gamma = gamma
        self._epsilon = epsilon  ## epsilon greey method
        self._timeout = timeout
        self._pi = None

        self._rng = np.random.RandomState(seed)

        def _get_state(self, state):
            """ the state space of VI is different from the state of the environment """
            print(state)
            return None #VI_state

        def __call__(self, state, verbose=False):
            self.value_iteration()
            plan = [state]
            for t in tqdm(range(self._max_num_steps)):
                action = self._get_action(state)
                state = self.T(state, action)
                plan.append(state)
                if self._check_goal(state):
                    print(f'!!! found goal in {t} steps!')
                    break

            return plan, {'node_expansions': 0}

        def value_iteration(self):
            """ performs value iteration and uses state action value to extract greedy policy for gridworld """

            start = time.time()

            # helper function
            def best_action(s):
                V_max = -np.inf
                for a in self._actions:
                    s_p = self.T(self._state_to_obs(self._states[s]), self._actions[a])
                    V_temp = self.R(self._state_to_obs(self._states[s_p]), self._actions[a]) + gamma * V[s_p[0], s_p[1] // 90]
                    V_max = max(V_max, V_temp)
                return V_max

            ## initialize V to be 0
            V = np.zeros(len(self._states))

            ## apply Bellman Equation until converge
            iter = 0
            while True:
                iter += 1
                delta = 0
                for s in self._states:
                    v = V[s]
                    V[s] = best_action(s)  # helper function used
                    delta = max(delta, abs(v - V[s]))

                # termination condition
                if delta < epsilon: break

            ## extract greedy policy
            pi = np.zeros((len(self._states), len(self._actions)))
            Q_sa = np.zeros((len(self._states), len(self._actions)))
            for s in range(len(self._states)):
                Q = np.zeros((len(self._actions),))
                for a in range(len(self._actions)):
                    s_p = self.T(self._state_to_obs(self._states[s]), self._actions[a])
                    Q[a] = self.R(self._state_to_obs(self._states[s_p]), self._actions[a]) + gamma * V[s]
                    Q_sa[s, a] = Q[a]

                ## highest state-action value
                Q_max = np.amax(Q)

                ## collect all actions that has Q value very close to Q_max
                # pi[s[0], s[1]//90, :] = softmax((Q * (np.abs(Q - Q_max) < 10**-2) / 0.01).astype(int))
                pi[s, :] = np.abs(Q - Q_max) < 10 ** -3
                pi[s, :] /= np.sum(pi[s, :])

            return pi

        def _get_action(self, state):
            if self._rng.rand() < epsilon:
                return self._rng.choice(self._actions)

            ## ties are broken randomly
            return np.random.choice(len(self._actions), p=self._pi[state, :])
