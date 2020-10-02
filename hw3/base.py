from collections import namedtuple, defaultdict, deque
from itertools import count, product
from tabulate import tabulate
import abc
import copy
import numpy as np
import heapq as hq
import pddlgym
from pddlgym.structs import Predicate, State, Type, LiteralConjunction
from pddlgym.parser import PDDLProblemParser
import functools
import tempfile
import os
import pyperplan
import time

"""### Classes
First we define some convenient abstract classes for Approach, Planner, Heuristic, and Featurizer.
"""

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
        self._plan, info = self._planner(obs)
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

    def __init__(self, successor_fn, check_goal_fn, heuristic=None, timeout=100):
        self._get_successor_state = successor_fn
        self._check_goal = check_goal_fn
        self._heuristic = heuristic or (lambda s : 0)
        self._timeout = timeout
        self._actions = None
        
    def __call__(self, state, verbose=True):
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

"""### Heuristics"""

class PyperplanHeuristic(Heuristic):
    """Don't worry about this -- it's just infrastructure connecting one library (PDDLGym) to another (Pyperplan)
    """
    def __init__(self, heuristic_name, domain):
        super().__init__()
        self._heuristic_name = heuristic_name
        self._domain = domain
        self._heuristic = None
        self._heuristic_goal = None
        self._actions = None

    def __call__(self, node):
        if node.state.goal != self._heuristic_goal:
            self._heuristic = self._initialize_heuristic(node.state)
            self._heuristic_goal = node.state.goal
        return self._heuristic(node.state)

    def set_actions(self, actions):
        self._actions = actions

    def train(self, env):
        pass
    
    def seed(self, seed):
        pass

    def _create_pyperplan_problem(self, state):
        try:
            problem_file = self._create_problem_file(state)
            parser = pyperplan.Parser(self._domain.domain_fname, problem_file)
            pyperplan_domain = parser.parse_domain()
            pyperplan_problem = parser.parse_problem(pyperplan_domain)
        finally:
            try:
                os.remove(problem_file)
            except FileNotFoundError:
                pass
        return pyperplan_problem

    def _create_problem_file(self, state):
        filename = tempfile.NamedTemporaryFile(delete=False).name
        lits = state.literals
        if not self._domain.operators_as_actions:
            lits |= set(self._actions)
        PDDLProblemParser.create_pddl_file(
            filename, state.objects-set(self._domain.constants), lits, 
            "myproblem", self._domain.domain_name, state.goal, fast_downward_order=True)
        return filename

    def _initialize_heuristic(self, state, cache_maxsize=10000):
        pyperplan_problem = self._create_pyperplan_problem(state)
        task = pyperplan.grounding.ground(pyperplan_problem)
        heuristic = pyperplan.HEURISTICS[self._heuristic_name](task)

        @functools.lru_cache(cache_maxsize)
        def _call_heuristic(state):
            state = frozenset({lit.pddl_str() for lit in state.literals})
            state &= task.facts
            node = pyperplan.search.searchspace.make_root_node(state)
            h = heuristic(node)
            return h

        return _call_heuristic

"""### Registering approaches"""

def get_approach(name, env, planning_timeout=10):
    """Put new approaches here!
    """
    if name == "astar_uniform":
        planner = AStar(env.get_successor_state, env.check_goal, timeout=planning_timeout)
        return SearchApproach(planner=planner)
    
    if name == "astar_hmax":
        heuristic = PyperplanHeuristic("hmax", domain=env.domain)
        planner = AStar(env.get_successor_state, env.check_goal, heuristic=heuristic, timeout=planning_timeout)
        return SearchApproach(planner=planner)
    
    if name == "astar_hff":
        heuristic = PyperplanHeuristic("hff", domain=env.domain)
        planner = AStar(env.get_successor_state, env.check_goal, heuristic=heuristic, timeout=planning_timeout)
        return SearchApproach(planner=planner)
    
    if name == "astar_hadd":
        heuristic = PyperplanHeuristic("hadd", domain=env.domain)
        planner = AStar(env.get_successor_state, env.check_goal, heuristic=heuristic, timeout=planning_timeout)
        return SearchApproach(planner=planner)

    raise Exception(f"Unrecognized approach: {name}")

# Add your approach names here
approaches = [
    "astar_uniform",
    "astar_hmax",
    "astar_hff",
    "astar_hadd",
]

"""### Evaluation Pipeline
Here's all the code that you should need to evaluate your approaches.
"""

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

"""### Here's where the action happens"""

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
#     print("\n# Standard Deviations #")
#     print(tabulate(std_table, headers=columns))

