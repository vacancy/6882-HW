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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from hw2.base import get_approach, run_single_test, run_single_experiment

approaches = [
    "random",
    "astar_uniform",
    "supervised_policy",
    "supervised_heuristic",
]

def print_level(level, all_results, columns, STD=False):
    print(f"\n### LEVEL {level} ###")
    mean_table = [(a,) + tuple(np.mean(all_results[level][a], axis=0)) for a in sorted(all_results[level])]
    std_table = [(a,) + tuple(np.std(all_results[level][a], axis=0)) for a in sorted(all_results[level])]
    print("\n# Means #")
    print(tabulate(mean_table, headers=columns))
    if STD:
        print("\n# Standard Deviations #")
        print(tabulate(std_table, headers=columns))

def main():
    levels = list(range(1, 7))
    columns = ["Approach", "Train Time", "Duration", "Num Steps", "Num Nodes", "Successes"]

    all_results = {}
    for level in levels:
        print('\n=======================================')
        print('Testing on Level ', level)
        all_results[level] = {}
        train_env = pddlgym.make(f"PDDLSearchAndRescueLevel{level}-v0")
        test_env = pddlgym.make(f"PDDLSearchAndRescueLevel{level}Test-v0")
        for approach in approaches:
            print('    Testing on Approach ', approach)
            all_results[level][approach] = []
            model = get_approach(approach, test_env)
            results = run_single_experiment(model, train_env, test_env)
            for (train_dur, dur, num_steps, num_nodes, succ) in zip(*results):
                all_results[level][approach].append((train_dur, dur, num_steps, num_nodes, succ))
        print_level(level, all_results, columns)

    for level in sorted(all_results):
        print_level(level, all_results, columns)


if __name__ == '__main__':
    main()
