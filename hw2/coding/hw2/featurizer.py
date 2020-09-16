#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : featurizer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 09/14/2020
#
# Distributed under terms of the MIT license.

import numpy as np
from pddlgym.structs import Predicate, State, Type, LiteralConjunction

from .base import Featurizer

class TabularFeaturizer(Featurizer):
    """A tabular featurizer assigns a unique ID to each input.
    """
    def __init__(self, one_hot=False):
        self._one_hot = one_hot
        self._x_to_idx = {}
        self._idx_to_x = {}
        self._num_features = 0
        self._unknown_idx = None
        self._initialized = False

    def initialize(self, all_data, DEBUG = False):
        for i, x in enumerate(sorted(set(all_data))):
            self._x_to_idx[x] = i
            self._idx_to_x[i] = x
        self._num_features = max(self._idx_to_x) + 1
        self._unknown_idx = self._num_features
        self._initialized = True
        if DEBUG: print(f"Initialized {self._num_features} tabular features")

    def apply(self, x):
        assert self._initialized, "Must call `initialize(all_data)` before `apply(datum)`."
        x_id = self._x_to_idx.get(x, self._unknown_idx)
        if self._one_hot:
            xhat = np.zeros(self._num_features + 1, dtype=np.float32)
            xhat[x_id] = True
            return xhat
        return x_id

    def invert(self, xhat):
        if self._one_hot:
            assert sum(xhat) == 1
            idx = np.argwhere(xhat)
        else:
            idx = xhat
        return self._idx_to_x[idx]


class PropositionalFeaturizer(Featurizer):
    """A propositional featurizer creates a boolean vector with one dimension per fact (Literal).
    """
    def __init__(self):
        self._x_to_idx = {}
        self._idx_to_x = {}
        self._num_features = 0
        self._initialized = False

    @classmethod
    def _wrap_goal_literal(cls, x):
        if isinstance(x, Predicate):
            return Predicate("WANT"+x.name, x.arity, var_types=x.var_types,
                is_negative=x.is_negative, is_anti=x.is_anti)
        new_predicate = cls._wrap_goal_literal(x.predicate)
        return new_predicate(*x.variables)

    def _preproc_pddl_state(self, X):
        if isinstance(X, State):
            return X.literals | {self._wrap_goal_literal(x) for x in X.goal.literals}
        return X

    def initialize(self, all_data, DEBUG = False):
        all_props = { x for X in all_data for x in self._preproc_pddl_state(X) }
        for i, x in enumerate(sorted(all_props)):
            self._x_to_idx[x] = i
            self._idx_to_x[i] = x
        self._num_features = max(self._idx_to_x)+1
        self._initialized = True
        if DEBUG: print(f"Initialized {self._num_features} propositional features")

    def apply(self, X):
        assert self._initialized, "Must call `initialize(all_data)` before `apply(datum)`."
        X = self._preproc_pddl_state(X)
        vec = np.zeros(self._num_features, dtype=np.float32)
        for x in X:
            try:
                idx = self._x_to_idx[x]
            except KeyError:
                continue
            vec[idx] = 1
        return vec

    def invert(self, vec):
        return { self._idx_to_x[idx] for idx in np.argwhere(vec) }


class SARStateFeaturizer(Featurizer):
    """This featurizer is specific to Search And Rescue states.
    It gives the dictionary-like state features that we saw in the previous homework.
    """
    def initialize(self, all_data):
        pass

    @classmethod
    def apply(cls, internal_state):
        state = { "carrying" : None }
        state["rescue"] = set()
        for lit in internal_state.goal.literals:
            state["rescue"].add(lit.variables[0].name)
        state["rescue"] = frozenset(state["rescue"]) # make hashable
        for lit in internal_state.literals:
            if lit.predicate.name.endswith("at"):
                obj_name = lit.variables[0].name
                r, c = cls._loc_to_rc(lit.variables[1])
                state[obj_name] = (r, c)
            if lit.predicate.name == "carrying":
                person_name = lit.variables[1].name
                state["carrying"] = person_name
        state = tuple(sorted(state.items())) # make hashable
        return state

    @staticmethod
    def _loc_to_rc(loc_str):
        assert loc_str.startswith("f") and loc_str.endswith("f")
        r, c = loc_str[1:-1].split('-')
        return (int(r), int(c))


class SARMinimalStateFeaturizer(Featurizer):
    """This featurizer is specific to Search And Rescue states.
    It puts the positions of the robot, hospital, and people into a normalized
    vector and ignores the walls. It also includes bits for whether each person
    is being carried and whether each person needs rescue.
    """
    # for normalization
    max_location = 6

    def initialize(self, all_data):
        pass

    def apply(self, x):
        sar_state = dict(SARStateFeaturizer.apply(x))
        state = []
        # add robot position
        state.extend(sar_state["robot0"])
        # add hospital position
        state.extend(sar_state["hospital0"])
        # get people
        people = sorted({ k for k in sar_state if k.startswith("person")})
        if sar_state["carrying"]:
            people.append(sar_state["carrying"])
            people.sort()
        # for each person...
        for person in people:
            # check whether the person is being carried
            if sar_state.get("carrying", None) == person:
                # add whether the person is being carried
                state.append(1.)
                # add the persons location (= robot's location)
                state.extend(sar_state["robot0"])
            else:
                # add whether the person is being carried
                state.append(0.)
                # add the persons location
                state.extend(sar_state[person])
            # add whether the person needs rescue
            state.append(float(person in sar_state["rescue"]))
        # normalize
        state = np.array(state, dtype=np.float32)
        state = (state / self.max_location) - 0.5
        return state

def get_featurizer(name):
    """Put new featurizers here!
    """
    if name == "tabular":
        return TabularFeaturizer()

    if name == "propositional":
        return PropositionalFeaturizer()

    if name == 'SARState':
        return SARStateFeaturizer()

    if name == 'SARMinimalState':
        return SARMinimalStateFeaturizer()

    raise Exception(f"Unrecognized featurizer: {name}")