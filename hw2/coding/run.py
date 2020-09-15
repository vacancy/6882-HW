#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : run.py
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

from hw2.base import get_approach

def run_single_test(test_env, problem_idx, model, max_horizon=250, max_duration=10):
    print(f"Running test problem {problem_idx} in environment {test_env.spec.id}")
    test_env.fix_problem_index(problem_idx)
    start_time = time.time()
    obs, info = test_env.reset()
    model_info = model.reset(obs)
    node_expansions = model_info.get('node_expansions', 0)
    num_steps = 0
    success = False
    for t in range(max_horizon):
        if time.time() - start_time > max_duration:
            break
        print(".", end='', flush=True)
        act = model.step(obs)
        obs, reward, done, info = test_env.step(act)
        num_steps += 1
        if done:
            assert reward == 1
            success = True
            break
    duration = time.time() - start_time
    print(f" final duration: {duration} with num steps {num_steps} and success={success}.")
    return duration, num_steps, node_expansions, success

def run_single_experiment(model, train_env, test_env, seed=0):
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

    for problem_idx in range(len(test_env.problems)):
        duration, num_steps, node_expansions, success = \
            run_single_test(test_env, problem_idx, model)
        test_durations.append(duration)
        test_num_steps.append(num_steps)
        test_node_expansions.append(node_expansions)
        test_successes.append(success)

    return train_durations, test_durations, test_num_steps, test_node_expansions, test_successes


approaches = [
    "random",
    "astar_uniform",
]

def main():
    levels = list(range(1, 7))

    all_results = {}
    for level in levels:
        all_results[level] = {}
        train_env = pddlgym.make(f"PDDLSearchAndRescueLevel{level}-v0")
        test_env = pddlgym.make(f"PDDLSearchAndRescueLevel{level}Test-v0")
        for approach in approaches:
            all_results[level][approach] = []
            model = get_approach(approach, test_env)
            results = run_single_experiment(model, train_env, test_env)
            for (train_dur, dur, num_steps, num_nodes, succ) in zip(*results):
                all_results[level][approach].append((train_dur, dur, num_steps, num_nodes, succ))


    columns = ["Approach", "Train Time", "Duration", "Num Steps", "Num Nodes", "Successes"]

    for level in sorted(all_results):
        print(f"\n### LEVEL {level} ###")
        mean_table = [(a, ) + tuple(np.mean(all_results[level][a], axis=0)) for a in sorted(all_results[level])]
        std_table = [(a, ) + tuple(np.std(all_results[level][a], axis=0)) for a in sorted(all_results[level])]
        print("\n# Means #")
        print(tabulate(mean_table, headers=columns))
        print("\n# Standard Deviations #")
        print(tabulate(std_table, headers=columns))


if __name__ == '__main__':
    main()
