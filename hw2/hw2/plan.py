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
import copy
import heapq as hq
import time
import matplotlib.pyplot as plt
from pddlgym.custom.searchandrescue import SearchAndRescueEnv

from .base import Approach
from .utils import display_image, draw_trace


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
            # print("Warning: step was called without a plan. Defaulting to random action.")
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
    def __init__(self, successor_fn, check_goal_fn,
                 num_search_iters=100, timeout=100,
                 replanning_interval=1, max_num_steps=250, gamma=0.9, seed=0):

        self._get_successor_state = successor_fn
        self._check_goal = check_goal_fn

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
        start_time = time.time()
        steps_since_replanning = 0
        plan = []
        for t in range(self._max_num_steps): #tqdm():
            if t % self._replanning_interval == 0:
                if verbose: print("Running UCT...")
                self.run(state, horizon=self._max_num_steps - t, verbose=verbose)
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
                if verbose: print("... UCT found goal at depth", t)
                break
            if time.time() - start_time > self._timeout:
                if verbose: print(f'.... stopped UCT (timeout = {self._timeout})')
                break

        return plan, {'node_expansions': len(self._Q)}

    def run(self, state, horizon=100, verbose=False):
        # Initialize Q[s][a][d] -> float
        self._Q = defaultdict(lambda : defaultdict(lambda : defaultdict(float)))
        # Initialize N[s][a][d] -> int
        self._N = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
        # Loop search
        start_time = time.time()
        for it in range(self._num_search_iters) : #tqdm(range(self._num_search_iters), desc='UCT search iterations'):
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
            # print('.... found reward at depth', depth)
            reward = 100
            q = reward
            # cost = 0
            # q = cost
        else:
            reward = 0
            q = reward + self._gamma * self._search(next_state, depth+1, horizon=horizon)
            # cost = 1
            # q = cost + self._gamma * self._search(next_state, depth+1, horizon=horizon)
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

    def seed(self, seed):
        self._rng = np.random.RandomState(seed)

class RTDP(Planner):
    """Implementation of RTDP based on Blai and Hector's 2003 paper
        Labeled RTDP: Improving the Convergence of Real-Time Dynamic Programming
    """
    def __init__(self, successor_fn, check_goal_fn, num_simulations=100,
                 epsilon = 0.01, timeout=10, max_num_steps=250, seed=0):

        self._get_successor_state = successor_fn
        self._check_goal = check_goal_fn

        self._num_simulations = num_simulations
        self._epsilon = epsilon
        self._max_num_steps = max_num_steps
        self._timeout = timeout
        self._rng = np.random.RandomState(seed)
        self._actions = None
        self._V = {}
        self._h = None

    def __call__(self, state, heuristic=None, verbose=False):
        self._h = heuristic
        plan, state_space_size = self.RTDP_trials(state)
        return plan, {'node_expansions': state_space_size}

    def set_actions(self, actions):
        self._actions = actions

    def _get_h(self, state):
        if self._h != None:
            return self._h(state)
        # if self._check_goal(state):
        #     return 1
        return 0

    def _get_V(self, state):
        if state not in self._V:
            self._V[state] = self._get_h(state)
        return self._V[state]

    def _get_Q(self, state, a):
        cost = 1
        if self._check_goal(state): cost = 0
        return cost + self._get_V(self._get_successor_state(state, a))

    def _update_q(self, state, action):
        # print('... updating ...', self._V[state], self._get_Q(state, action))
        self._V[state] = self._get_Q(state, action)

    def _get_greedy_action(self, state):
        return min(self._actions, key=lambda a : (self._get_Q(state, a), self._rng.uniform()))

    def RTDP_trials(self, initial_state, verbose=False):
        ## update Values through simulations
        start_time = time.time()
        found_goal = 1000
        actions = []
        for i in range(self._num_simulations): ## tqdm(range(self._num_simulations), desc='RTDP simulations'):
            last_V = copy.deepcopy(self._V)
            state = initial_state
            for t in range(self._max_num_steps):
                action = self._get_greedy_action(state)
                actions.append(action)
                self._update_q(state, action)
                state = self._get_successor_state(state, action)
                # print(len(self._V), max(self._V.values()), sum(self._V.values())/len(self._V))
                if self._check_goal(state):
                    if t != found_goal:
                        found_goal = t
                        if verbose: print(f'.... found goal in {i}-th simulation after {t} steps') # actions
                    break
                if time.time() - start_time > self._timeout:
                    if verbose: print(f'.... stopped RTDP after {t} iterations (timeout = {self._timeout})')
                    break
            if time.time() - start_time > self._timeout:
                break
            converged = len(last_V) == len(self._V)
            for s in last_V:
                if abs(self._get_V(s) - last_V[s]) > self._epsilon:
                    # if max(self._V.values()) == 1: print('____',self._get_V(s), last_V[s])
                    converged = False
            if converged:
                # print(f'.... RTDP converged after {i} simulations')
                break
            actions = []

        ## sequence of greedy actions extracted from converged V
        return actions, len(self._V)

    def _get_residual(self, state):
        # print(self._get_V(state), self._get_Q(state, self._get_greedy_action(state)))
        action = self._get_greedy_action(state)
        return abs(self._get_V(state) - self._get_Q(state, action))

class LRTDP(Planner):
    """Implementation of LRTDP based on Blai and Hector's 2003 paper
        Labeled RTDP: Improving the Convergence of Real-Time Dynamic Programming
    """
    def __init__(self, successor_fn, check_goal_fn, num_simulations=100,
                 epsilon = 0.01, timeout=10, max_num_steps=250, seed=0):

        super().__init__()
        self.wrapped = RTDP(successor_fn, check_goal_fn, num_simulations,
                 epsilon, timeout/2, max_num_steps, seed)
        self._solved = {}
        self._timeout = timeout/2

    def __call__(self, state, heuristic=None, verbose=False):
        self.wrapped._h = heuristic
        _, state_space_size = self.LRTDP_trials(state)
        plan, _ = self.wrapped.RTDP_trials(state)
        return plan, {'node_expansions': state_space_size}

    def set_actions(self, actions):
        self._actions = actions
        self.wrapped._actions = actions

    def _get_solved(self, state):
        if state not in self._solved:
            self._solved[state] = False
        return self._solved[state]

    def _get_residual(self, state):
        return self.wrapped._get_residual(state)

    def _percentage_solved(self):
        count = 0
        for s in self._solved:
            if self._solved[s]:
                count += 1
        print(f'... {round(count/len(self._solved), 3)} out of {len(self._solved)} states are solved')

    def _check_solved(self, state):
        rv = True
        open = []
        closed = []
        if not self._get_solved(state):
            open.append(state)

        while len(open) > 0:
            s = open[len(open)-1]
            open.remove(s)
            closed.append(s)

            ## check residual
            residual = self._get_residual(s)
            if residual > self.wrapped._epsilon:
                # print(len(open), len(closed), list(self._solved.keys()).index(s), residual, self.wrapped._epsilon)
                rv = False
                continue

            ## expand state
            action = self.wrapped._get_greedy_action(s)
            next = self.wrapped._get_successor_state(s, action)
            # self._percentage_solved()
            if (not self._get_solved(next)) and (next not in open and next not in closed):
                open.append(next)

        if rv:
            ## label relevant states
            for s in closed:
                self._solved[s] = True
        else:
            ## update states with residuals and ancestors
            while len(closed) > 0:
                s = closed[len(closed) - 1]
                closed.remove(s)
                action = self.wrapped._get_greedy_action(s)
                self.wrapped._update_q(s, action)
        return rv

    def LRTDP_trials(self, initial_state, verbose=False):
        start_time = time.time()
        found_goal = 1000
        actions = []
        self._check_solved(initial_state)

        while not self._get_solved(initial_state):
            state = initial_state
            visited = []
            while not self._get_solved(state):
                visited.append(state)
                if self.wrapped._check_goal(state):
                    if len(actions) != found_goal:
                        found_goal = len(actions)
                        # print(f'.... found goal after {len(actions)} expansions')
                    break
                action = self.wrapped._get_greedy_action(state)
                actions.append(action)
                self.wrapped._update_q(state, action)
                state = self.wrapped._get_successor_state(state, action)

                if time.time() - start_time > self._timeout:
                    if verbose: print(f'.... stopped LRTDP (timeout = {self._timeout})')
                    break

            if time.time() - start_time > self._timeout:
                if verbose: print(f'.... stopped LRTDP (timeout = {self._timeout})')
                break

            ## try labeling visited states in reverse order
            while len(visited) > 0:
                s = visited[len(visited)-1]
                visited.remove(s)
                if not self._check_solved(s):
                    break

            actions = []

        # self._percentage_solved()
        return actions, len(self.wrapped._V)

class DPApproach(Approach):
    def __init__(self, planner):
        self._planner = planner
        self._actions = None
        self._plan = []

    def set_actions(self, actions):
        self._actions = actions
        self._planner.set_actions(actions)

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
    def __init__(self, env, successor_fn, check_goal_fn,
                 max_num_steps=150, gamma=0.9, epsilon=0, timeout=100, seed=0):

        self._env_ori = env
        self._env = SearchAndRescueEnv(env)
        self._states = None
        self._actions = None
        self.T = successor_fn
        self._check_goal = check_goal_fn

        self._max_num_steps = max_num_steps
        self._gamma = gamma
        self._epsilon = epsilon  ## epsilon greedy method
        self._timeout = timeout
        self._pi = None

        self._rng = np.random.RandomState(seed)

    def set_actions(self, actions):
        self._actions = actions

    def _obs_to_state(self, state):
        """ from the observation in the environment to the state space of VI """
        new_state = self._env._internal_to_state(state)
        if self._level == 1:
            goal_person = list([y for x,y in new_state if 'rescue' == x][0])[0]
            rob_loc = [y for x,y in new_state if 'robot' in x][0]
            person_loc = [y for x,y in new_state if goal_person == x]
            if len(person_loc) > 0: person_loc = person_loc[0]
            else: person_loc = rob_loc
            if_carry = [y for x, y in new_state if 'carrying' == x][0]
            if if_carry: if_carry = 1
            else: if_carry = 0
            return self._states.index((self._available_locations.index(rob_loc),
                    self._available_locations.index(person_loc),
                    if_carry))
        raise NotImplementedError("Didn't implement for this level!")

    def _state_to_obs(self, state):
        """ from the state space of VI to the observation in the environment """

        if self._level == 1:
            goal_person = list([y for x, y in self._fixed_state_items if 'rescue' == x][0])[0]
            rob_loc, person_loc, if_carry = self._states[state]
            if if_carry == 1: if_carry = goal_person
            else: if_carry = None
            new_state = copy.deepcopy(self._fixed_state_items)
            new_state.append(('robot0', self._available_locations[rob_loc]))
            new_state.append(('carrying', if_carry))
            if if_carry != goal_person:
                new_state.append((goal_person, self._available_locations[person_loc]))
            return self._env._state_to_internal(tuple(new_state))

        raise NotImplementedError("Didn't implement for this level!")

    def _set_states(self, state):

        ## the available states include robot-at x person-at x handsfree
        new_state = self._env._internal_to_state(state)
        self._available_locations = [(r,c) for r in range(6) for c in range(6)]
        self._fixed_state_items = [(x,y) for x,y in new_state
                                   if 'hospital' in x or 'wall' in x or 'rescue' == x]

        ## find the number of possible locations
        for item, value in new_state:
            if 'wall' in item:
                self._available_locations.remove(value)

        ## find the level specified by looking at ('rescue', frozenset({'personx'})) and ('personx')
        num_of_rescues = len(list([y for x,y in new_state if 'rescue' == x][0]))
        num_of_persons = len(list([y for x,y in new_state if 'person' in x]))

        ## level 1 or level 2
        if num_of_persons == 1 and num_of_rescues == 1:
            self._level = 1
            self._states = tuple([(rob_loc, person_loc, if_carry)
                                  for rob_loc in range(len(self._available_locations))
                                  for person_loc in range(len(self._available_locations))
                                  for if_carry in range(2)
                                  ])

            return self._states

        raise NotImplementedError("Didn't implement for this level!")

    def __call__(self, state, verbose=False): # self._env.render_from_state(self._env._internal_to_state(state))
        display_image(self._env_ori.render_from_state(state), 'initial')

        self._set_states(state)
        self._pi = self.value_iteration()
        plan = []
        states = [state]
        for t in tqdm(range(self._max_num_steps)):
            action = self._get_action(state, t)
            plan.append(action)
            state = self.T(state, action)
            states.append(state)
            # if self._obs_to_state(state) in [1798, 1799]:
                # print(self._env._internal_to_state(state))
                # print(self._check_goal(state), state)
                # s = self._state_to_obs(self._obs_to_state(state))
                # print(self._check_goal(s), s)
                # print()
            if self._check_goal(state):
                print(f'!!! found goal in {t+1} steps!')
                break

        draw_trace(self._translate_plan(states))
        plt.savefig('test.png')
        return plan, {'node_expansions': len(self._states)}

    def _translate_plan(self, plan):
        new_plan = []
        numbered_plan = []
        for obs in plan:
            new_plan.append(self._env._internal_to_state(obs))
            numbered_plan.append(self._obs_to_state(obs))
        # print(numbered_plan)
        # print(len(new_plan))
        return new_plan

    def value_iteration(self):
        """ performs value iteration and uses state action value to extract greedy policy for gridworld """

        start = time.time()

        ## save the value as a dictionary as it will be used repeatedly
        T = {}
        def get_T(s, a):
            if (s, a) not in T:
                T[(s, a)] = self._obs_to_state(self.T(self._state_to_obs(s), self._actions[a]))
            return T[(s, a)]

        R = {}
        def get_R(s, a):
            if (s, a) not in R:
                # R[(s, a)] = self.R(self._state_to_obs(s), self._actions[a])
                # if self._check_goal(self._state_to_obs(get_T(s, a))):
                if self._check_goal(self._state_to_obs(s)):
                    R[(s, a)] = 100
                else:
                    R[(s, a)] = 0
            return R[(s, a)]

        # helper function
        def best_action(s):
            V_max = -np.inf
            for a in range(len(self._actions)):
                s_p = get_T(s, a)
                V_temp = get_R(s, a) + self._gamma * V[s_p]
                V_max = max(V_max, V_temp)
            return V_max

        ## initialize V to be 0
        V = np.zeros(len(self._states))

        ## apply Bellman Equation until converge
        iter = 0
        while True:
            iter += 1
            delta = 0
            for s in range(len(self._states)):
                v = V[s]
                V[s] = best_action(s)  # helper function used
                delta = max(delta, abs(v - V[s]))

            if iter == 1:
                print(f'... finished translating {len(self._states)} states in {round(time.time() - start, 3)} seconds')
                start = time.time()

            # termination condition
            if delta < 10**-2 or time.time()-start > self._timeout: break
            # else: print('... ', delta)

        ## extract greedy policy
        pi = np.zeros((len(self._states), len(self._actions)))
        Q_sa = np.zeros((len(self._states), len(self._actions)))
        for s in range(len(self._states)):
            Q = np.zeros((len(self._actions),))
            for a in range(len(self._actions)):
                s_p = get_T(s, a)
                Q[a] = get_R(s, a) + self._gamma * V[s_p]
                Q_sa[s, a] = Q[a]

            ## highest state-action value
            Q_max = np.amax(Q)

            ## collect all actions that has Q value very close to Q_max
            # pi[, :] = softmax((Q * (np.abs(Q - Q_max) < 10**-2) / 0.01).astype(int))
            pi[s, :] = np.abs(Q - Q_max) < 10 ** -3
            pi[s, :] /= np.sum(pi[s, :])

        print(f'... finished VI on {len(self._states)} states in {round(time.time()-start, 3)} seconds')
        return pi

    def _get_action(self, obs, t=None):
        if self._rng.rand() < self._epsilon:
            act = self._rng.choice(self._actions)
            print('------ chosen random action', act)
            return act

        ## ties are broken randomly
        state = self._obs_to_state(obs)
        action_chosen = self._actions[np.random.choice(len(self._actions), p=self._pi[state, :])]
        # print(t, self._obs_to_state(obs), self._pi[state, :])
        return action_chosen