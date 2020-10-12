import abc
import pddlgym
import heapq as hq
import numpy as np
import time
import random
from itertools import count
from collections import defaultdict, namedtuple
from tabulate import tabulate
from .plan import UCT, RTDP, LRTDP

"""## A Generic Approach for PO Environments

An agent maintains a belief state and produces actions.
"""

class PartialObservabilityApproach:
    """An agent that maintains a belief state (set of possible states)
    as it takes actions and receives partial observations.

    Parameters
    ----------
    actions : [ int ]
        A list of actions that the agent can take. All actions are
        applicable in all states.
    successor_fn : state, action -> state
        Maps an environment state and action to a next state.
    check_goal_fn : state -> bool
        Maps an environment state to true when the goal is reached.
    observation_fn : state -> observation
        Maps an environment state to an observation. Sometimes
        called "Percept".
    observation_to_states_fn : observation -> frozenset{states}
        Maps an observation to the set of environment states such
        that observation_fn(state) would produce that observation.
    """
    def __init__(self, actions, successor_fn, check_goal_fn,
                 observation_fn, observation_to_states_fn):
        self._actions = actions
        self._successor_fn = successor_fn
        self._check_goal_fn = check_goal_fn
        self._observation_fn = observation_fn
        self._observation_to_states_fn = observation_to_states_fn
        self._step_count = 0
        self._belief_state = None # set after reset
        self._rng = None

    def set_actions(self, actions):
        self._actions = actions

    def reset(self, obs):
        """Tell the agent that we have started a new problem with
        initial observation "obs".

        Parameters
        ----------
        obs : hashable
            The initial observation

        Returns
        -------
        info : dict
            Any useful debug info.
        """
        # Reset the belief state
        self._belief_state = self._observation_to_states_fn(obs)
        # Reset step count
        self._step_count = 0
        return {}

    def step(self, obs):
        """Receive an observation and produce an action to be
        immediately executed in the environment.

        Parameters
        ----------
        obs : hashable
            The observation

        Returns
        -------
        action : int
            The action is assumed to be immediately taken.
        info : dict
            Any useful debug info.
        """
        # Update the belief state based on the observation
        possible_states = self._observation_to_states_fn(obs)
        # This is set intersection
        self._belief_state &= possible_states
        # Find an action
        action, info = self._get_action()
        # Update step count
        self._step_count += 1
        # Update the belief state based on action
        self._belief_state = self._predict_belief_state(self._belief_state,
            action)
        return action

    def train(self, env):
        pass

    def seed(self, seed):
        """Seed the agent, just in case it's random
        """
        self._rng = np.random.RandomState(seed)

    @abc.abstractmethod
    def _get_action(self):
        """Return an action to be immediately taken, based on the current
        belief state (self._belief_state). This is the main thing that
        differentiates subclasses.

        Returns
        -------
        action : int
            The action to be taken immediately.
        info : dict
            Any useful debug info.
        """
        raise NotImplementedError("Override me")

    def _check_belief_state_goal(self, belief_state):
        """Check whether the belief state is a goal, that is, whether
        all states in the belief state satisfy the check_goal_fn.

        This function is included here because it is likely to be
        used by subclasses.

        Parameters
        ----------
        belief_state : frozenset{hashable}

        Returns
        -------
        goal_reached : bool
        """
        # We've found a goal if all states in the belief state are at goals
        for state in belief_state:
            if not self._check_goal_fn(state):
                return False
        return True

    def _predict_belief_state(self, belief_state, action):
        """Get the next belief state that would result after taking
        action in belief_state.

        This function is included here because it is likely to be
        used by subclasses.

        Parameters
        ----------
        belief_state : frozenset{hashable}
        action : int

        Returns
        -------
        next_belief_state : frozenset{hashable}
        """
        next_belief_state = set()
        for state in belief_state:
            next_state = self._successor_fn(state, action)
            next_belief_state.add(next_state)
        return frozenset(next_belief_state)

class AndOrSearch(PartialObservabilityApproach):
    """Exhaustive depth-first And-Or search (Russell-Norvig version)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conditional_plan = "failure"
        self._num_expansions = None # set in reset

    def _get_action(self):
        """Return an action to be immediately taken, based on the current
        belief state (self._belief_state).

        Returns
        -------
        action : int
            The action to be taken immediately.
        info : dict
            Any useful debug info.
        """
        info = {}
        # Check if it's time to replan
        if self._step_count == 0 or self._conditional_plan == "failure":
            action, self._conditional_plan = self._get_conditional_plan()
            info["node_expansions"] = next(self._num_expansions)-1
            self._num_expansions = None
        else:
            action, self._conditional_plan = self._conditional_plan[self._belief_state]
        return action, info

    def _get_conditional_plan(self):
        """Run planning from scratch, given the current belief state
        """
        # Reset expansion counter
        self._num_expansions = count()
        # Start off the AND-OR search
        return self._run_or_search(self._belief_state, [])

    def _run_or_search(self, belief_state, path, depth=0, max_depth=np.inf, verbose=True):
        """Run OR part of AO search (recursively).

        Parameters
        ----------
        belief_state : frozenset{hashable}
        path : [ belief_state ]
            Belief states encountered so far, used to find cycles.
        depth : int
        max_depth : int

        Returns
        -------
        conditional_plan : Any
            Representation of the conditional plan.
        """
        # Max depth exceeded
        if depth > max_depth:
            return "failure"
        if self._check_belief_state_goal(belief_state):
            return "success"
        # Check if we've visited this belief state already
        if belief_state in path:
            return "failure"
        # Consider each possible action, depth-first order
        num_expansions = next(self._num_expansions)
        if verbose:
            print(f"Expanding OR  node {num_expansions}", end='\r', flush=True)
        for action in self._actions:
            next_belief_states = self._get_belief_successor_states(belief_state, action)
            conditional_plan = self._run_and_search(next_belief_states, [belief_state] + path,
                depth=depth, max_depth=max_depth, verbose=verbose)
            if conditional_plan != "failure":
                return (action, conditional_plan)
        return "failure"

    def _run_and_search(self, belief_states, path, depth=0, max_depth=np.inf, verbose=True):
        """Run AND part of the AO search (recursively).

        Parameters
        ----------
        belief_states : [frozenset{hashabale}]
            A list of belief states.
        path : [ belief_state ]
            Belief states encountered so far, used to find cycles.
        depth : int
        max_depth : int

        Returns
        -------
        conditional_plan : Any
            Representation of the conditional plan.
        """
        plans = []
        num_expansions = next(self._num_expansions)
        if verbose:
            print(f"Expanding AND node {num_expansions}", end='\r', flush=True)
        for belief_state in belief_states:
            plan = self._run_or_search(belief_state, path, depth=depth+1,
                                       max_depth=max_depth, verbose=verbose)
            if plan == "failure":
                return "failure"
            plans.append(plan)
        return dict(zip(belief_states, plans))

    def _get_belief_successor_states(self, belief_state, action):
        """Get the next possible belief states based on possible
        observations, after taking action in belief_state.

        Parameters
        ----------
        belief_state : frozenset{hashable}
        action : int

        Returns
        -------
        next_belief_states : [frozenset{hashabale}]
            A list of belief states.
        """
        # Prediction
        next_belief_state = self._predict_belief_state(belief_state, action)
        # Possible percepts and update
        possible_percept_to_states = defaultdict(list)
        for s in next_belief_state:
            o = self._observation_fn(s)
            possible_percept_to_states[o].append(s)
        # Sorting to ensure determinism
        belief_states = sorted(frozenset(bs) for bs in possible_percept_to_states.values())
        return belief_states


class IterativeDeepeningAndOrSearch(AndOrSearch):
    """Run AndOrSearch with progressively larger depth limits until a plan is found.
    """
    def _get_conditional_plan(self):
        """Run planning from scratch, given the current belief state
        """
        # Reset expansion counter
        self._num_expansions = count()
        # Run iterative deepening planning until plan is not a failure
        for max_depth in count(1):
            print(f"Running iterative deepening with depth {max_depth}", end='\r', flush=True)
            conditional_plan = self._run_or_search(self._belief_state, [],
                depth=0, max_depth=max_depth, verbose=False)
            if conditional_plan != "failure":
                print()
                break
        return conditional_plan

"""### PO-UCT
Finish implementing this class.
"""

class POUCT(PartialObservabilityApproach):
    """Use UCT in belief space; sample belief state transitions uniformly at random
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._planner = UCT(successor_fn=self._get_uct_transition,
                            check_goal_fn=lambda s : self._check_belief_state_goal(s),
                            num_search_iters=100, gamma=0.9)
        self._steps_since_replanning = 0
        self._replanning_interval = 1
        self._horizon = 50

    def set_actions(self, actions):
        self._actions = actions
        self._planner.set_actions(actions)

    def _get_action(self):
        """Return an action to be immediately taken, based on the current
        belief state (self._belief_state).

        Returns
        -------
        action : int
            The action to be taken immediately.
        info : dict
            Any useful debug info.
        """
        info = {}
        # Replan on a fixed interval
        if self._step_count % self._replanning_interval == 0:
            info = self._planner.run(self._belief_state, horizon=self._horizon)
            self._steps_since_replanning = 0
        action = self._planner._get_action(self._belief_state, t=self._steps_since_replanning)
        self._steps_since_replanning += 1
        return action, info

    def _get_plan(self):
        """Determinize and plan

        Returns
        -------
        plan : [ int ]
            A sequence of actions.
        """
        self._planner.run(self._belief_state)
        return self._planner._get_action(self._belief_state)

    def _get_uct_reward(self, belief_state, _):
        """Use a sparse reward: 1.0 if the goal is reached, 0 otherwise

        Parameters
        ----------
        belief_state : frozenset{hashable}

        Returns
        -------
        reward : float
        """
        if self._check_belief_state_goal(belief_state): return 1.0
        return 0

    def _get_uct_transition(self, belief_state, action):
        """Sample uniformly at random among the possible next belief states
        """
        next_b = self._predict_belief_state(belief_state, action)
        sampled_b = random.choice(list(next_b))
        return frozenset({sampled_b})

    def seed(self, seed):
        """Also seed the planner
        """
        super().seed(seed)
        self._planner.seed(seed)


class PORTDP(PartialObservabilityApproach):
    """Use UCT in belief space; sample belief state transitions uniformly at random
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._planner = RTDP(successor_fn=self._get_transition,
                            check_goal_fn=lambda s : self._check_belief_state_goal(s),
                            num_simulations=100, max_num_steps=50)

    def set_actions(self, actions):
        self._actions = actions
        self._planner.set_actions(actions)

    def _get_action(self):
        """Return an action to be immediately taken, based on the current
        belief state (self._belief_state).

        Returns
        -------
        action : int
            The action to be taken immediately.
        info : dict
            Any useful debug info.
        """
        plan, info = self._planner(self._belief_state)
        if len(plan) == 0:
            return self._rng.choice(self._actions), info
        return plan[0], info

    def _get_plan(self):
        """Determinize and plan

        Returns
        -------
        plan : [ int ]
            A sequence of actions.
        """
        plan, info = self._planner(self._belief_state)
        return plan

    def _get_transition(self, belief_state, action):
        """Sample uniformly at random among the possible next belief states
        """
        next_b = self._predict_belief_state(belief_state, action)
        sampled_b = random.choice(list(next_b))
        return frozenset({sampled_b})

    def seed(self, seed):
        """Also seed the planner
        """
        super().seed(seed)


class POLRTDP(PartialObservabilityApproach):
    """Use UCT in belief space; sample belief state transitions uniformly at random
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._planner = LRTDP(successor_fn=self._get_transition,
                            check_goal_fn=lambda s : self._check_belief_state_goal(s),
                            num_simulations=100, max_num_steps=50)

    def set_actions(self, actions):
        self._actions = actions
        self._planner.set_actions(actions)

    def _get_action(self):
        """Return an action to be immediately taken, based on the current
        belief state (self._belief_state).

        Returns
        -------
        action : int
            The action to be taken immediately.
        info : dict
            Any useful debug info.
        """
        plan, info = self._planner(self._belief_state)
        return plan[0], info

    def _get_plan(self):
        """Determinize and plan

        Returns
        -------
        plan : [ int ]
            A sequence of actions.
        """
        plan, info = self._planner(self._belief_state)
        return plan

    def _get_transition(self, belief_state, action):
        """Sample uniformly at random among the possible next belief states
        """
        next_b = self._predict_belief_state(belief_state, action)
        sampled_b = random.choice(list(next_b))
        return frozenset({sampled_b})

    def seed(self, seed):
        """Also seed the planner
        """
        super().seed(seed)