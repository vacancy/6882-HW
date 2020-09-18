#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : base.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/14/2020
#
# Distributed under terms of the MIT license.

import abc
import time
from tqdm import tqdm


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

def get_states(env):
    print('getting states')

def get_approach(name, env, planning_timeout=10, num_search_iters=1000, gamma=0.999):
    """Put new approaches here!
    """
    if name == "random":
        from .plan import RandomActions
        return RandomActions()

    if name == "astar_uniform":
        from .plan import SearchApproach, AStar
        planner = AStar(env.get_successor_state, env.check_goal, timeout=planning_timeout)
        return SearchApproach(planner=planner)

    if name == "uct":
        from .plan import SearchApproach, UCT
        planner = UCT(env.get_successor_state, env.check_goal, reward_fn=env.extrinsic_reward,
                      num_search_iters=num_search_iters, timeout=planning_timeout/10,
                      replanning_interval=1, max_num_steps=100, gamma=gamma)
        return SearchApproach(planner=planner)

    if name == 'value_iteration':
        from .plan import DPApproach, VI
        planner = VI(env, env.get_successor_state, env.check_goal, reward_fn=env.extrinsic_reward,
                     max_num_steps=100, gamma=gamma)
        return DPApproach(planner=planner)

    if name == 'supervised_policy':
        from .learning import SupervisedPolicyLearning
        return SupervisedPolicyLearning()

    if name == 'supervised_heuristic':
        from .learning import SupervisedHeuristicLearning, LearnedHeuristic
        from .plan import SearchApproach, AStar
        planner = AStar(env.get_successor_state, env.check_goal, timeout=planning_timeout)
        return SupervisedHeuristicLearning(planner, LearnedHeuristic())

    if name == 'qlearning_heuristic':
        from .learning import SupervisedHeuristicLearning, QLearningHeuristic
        from .plan import SearchApproach, AStar
        planner = AStar(env.get_successor_state, env.check_goal, timeout=planning_timeout)
        return SupervisedHeuristicLearning(planner, QLearningHeuristic(
            H=1000,       # horizon,
            T=int(1e5),   # total env steps
            gamma=0.9,    # gamma
            epsilon=0.1,  # in eps-greedy
        ))

    raise Exception(f"Unrecognized approach: {name}")


def run_single_test(test_env, problem_idx, model, max_horizon=250, max_duration=10, DEBUG=False):
    if DEBUG: print(f"Running test problem {problem_idx} in environment {test_env.spec.id}")
    test_env.fix_problem_index(problem_idx)
    start_time = time.time()
    obs, info = test_env.reset()
    model_info = model.reset(obs)
    node_expansions = model_info.get('node_expansions', 0)
    num_steps = 0
    success = False
    states, actions = [obs], []
    last_action = None
    for t in range(max_horizon):
        if time.time() - start_time > max_duration:
            break
        if DEBUG: print(".", end='', flush=True)
        act = model.step(obs)
        last_action = act
        obs, reward, done, info = test_env.step(act)

        num_steps += 1
        if done:
            assert reward == 1
            success = True
            break
        else:
            ## collect states and actions data
            actions.append(act)
            states.append(obs)

    actions.append(last_action)

    duration = time.time() - start_time
    if DEBUG: print(f" final duration: {duration} with num steps {num_steps} and success={success}.")
    return duration, num_steps, node_expansions, success, states, actions

def run_single_experiment(model, train_env, test_env, seed=0, num_problems=100):
    # Initialize
    test_env.reset()
    actions = test_env.get_possible_actions()
    model.set_actions(actions)
    model.seed(seed)

    # Training
    training_start_time = time.time()
    model.train(train_env)
    train_duration = time.time() - training_start_time
    train_durations = [train_duration] * len(test_env.problems) # for result reporting convenience

    # Test time
    test_durations = [] # seconds, one per problem
    test_num_steps = [] # integers
    test_node_expansions = [] # integers
    test_successes = [] # boolean, True if successful

    num_problems = min(len(test_env.problems), num_problems)
    for problem_idx in tqdm(range(num_problems)):
        duration, num_steps, node_expansions, success, _, _ = \
            run_single_test(test_env, problem_idx, model)
        test_durations.append(duration)
        test_num_steps.append(num_steps)
        test_node_expansions.append(node_expansions)
        test_successes.append(success)
    print()
    return train_durations, test_durations, test_num_steps, test_node_expansions, test_successes
