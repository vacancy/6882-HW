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
import csv
from os.path import join
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from envs.water_delivery import WaterDeliveryEnv

from hw2.base import get_approach, run_single_test, run_single_experiment

env = ['sar','water','posar'][1]

approaches = [

    # "random",
    # "astar_uniform",
    # "uct1",
    # "uct2",
    # "uct3",
    # "uct4",
    # "uct5",
    # "uct6",
    # "lrtdp",
    # "rtdp",

    # "dfaos",
    # "idaos",
    # "pouct",
    "po-lrtdp",
    "po-rtdp",

    # "value_iteration",
    # "supervised_policy",
    # "supervised_heuristic",
    # "qlearning_heuristic",
]
levels = {
    'sar': list(range(1, 5)),
    'water': ['default', 'medium', 'hard'],
    'posar': [
        "SmallPOSARRadius0",
        "POSARRadius1",
        "POSARRadius0",
        "POSARRadius1Xray",
        "POSARRadius0Xray",
    ],
}[env]
columns = ["Approach", "Train Time", "Duration", "Num Steps", "Num Nodes", "Successes"]


def print_level(level, all_results, columns, STD=False, CSV=True, CSV_PREFIX=''):
    print(f"\n### LEVEL {level} ###")
    mean_table = [(a,) + tuple(np.mean(all_results[level][a], axis=0)) for a in sorted(all_results[level])]
    std_table = [(a,) + tuple(np.std(all_results[level][a], axis=0)) for a in sorted(all_results[level])]
    print("\n# Means #")
    print(tabulate(mean_table, headers=columns))
    if STD:
        print("\n# Standard Deviations #")
        print(tabulate(std_table, headers=columns))
    if CSV:
        with open(join('results', f'means_{CSV_PREFIX}{level}.csv'), 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            # wr.writerow([f'LEVEL {level}'])
            wr.writerow(columns)
            for row in mean_table:
                wr.writerow(list(row))
            wr.writerow('')
        with open(join('results', f'std_{CSV_PREFIX}{level}.csv'), 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            # wr.writerow([f'LEVEL {level}'])
            wr.writerow(columns)
            for row in std_table:
                wr.writerow(list(row))
            wr.writerow('')

def get_env(env, level):
    if env == 'sar':
        train_env = pddlgym.make(f"PDDLSearchAndRescueLevel{level}-v0")
        test_env = pddlgym.make(f"PDDLSearchAndRescueLevel{level}Test-v0")
    elif env == 'water':
        train_env = WaterDeliveryEnv(mode=level)
        test_env = WaterDeliveryEnv(mode=level)
    elif env == 'posar':
        train_env = pddlgym.make(f"{level}-v0")
        test_env = pddlgym.make(f"{level}-v0")
    return train_env, test_env

def main():
    all_results = {}
    for level in levels:
        print('\n=======================================')
        print('Experimenting on Level:', level)
        all_results[level] = {}
        train_env, test_env = get_env(env, level)
        for approach in approaches:
            print('    Experimenting with Approach:', approach)
            all_results[level][approach] = []
            model = get_approach(approach, test_env)
            results = run_single_experiment(model, train_env, test_env, num_problems=100)
            for (train_dur, dur, num_steps, num_nodes, succ) in zip(*results):
                all_results[level][approach].append((train_dur, dur, num_steps, num_nodes, succ))
        print_level(level, all_results, columns) ## , CSV_PREFIX='PO_'

    for level in sorted(all_results):
        print_level(level, all_results, columns, STD=False, CSV=False)


if __name__ == '__main__':
    main()

