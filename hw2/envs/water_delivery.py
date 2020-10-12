try:
    from .utils import render_from_layout, get_asset_path
except ImportError:
    from utils import render_from_layout, get_asset_path
import functools
import numpy as np
import matplotlib.pyplot as plt


class WaterDeliveryEnv:
    """A grid world where a robot must pick up a water bottle and then
    deliver water to all people in the grid. Rewards are given when each
    person is quenched.

    Parameters
    ----------
    layout : np.ndarray, layout.shape = (height, width, num_objects)
        The initial state.
    """
    # Types of objects
    OBJECTS = ROBOT, ROBOT_WITH_WATER, WATER, PERSON, QUENCHED_PERSON = range(5)

    # Create layouts
    DEFAULT_LAYOUT = np.zeros((5, 5, len(OBJECTS)), dtype=bool)
    DEFAULT_LAYOUT[4, 2, ROBOT] = 1
    DEFAULT_LAYOUT[0, 4, WATER] = 1
    DEFAULT_LAYOUT[1, 0, PERSON] = 1
    DEFAULT_LAYOUT[2, 0, PERSON] = 1
    DEFAULT_LAYOUT[3, 0, PERSON] = 1

    MEDIUM_LAYOUT = np.zeros((9, 9, len(OBJECTS)), dtype=bool)
    MEDIUM_LAYOUT[8, 4, ROBOT] = 1
    MEDIUM_LAYOUT[0, 0, WATER] = 1
    MEDIUM_LAYOUT[1, 0, PERSON] = 1
    MEDIUM_LAYOUT[5, 5, PERSON] = 1
    MEDIUM_LAYOUT[3, 4, PERSON] = 1
    MEDIUM_LAYOUT[4, 3, PERSON] = 1
    MEDIUM_LAYOUT[8, 7, PERSON] = 1
    MEDIUM_LAYOUT[6, 0, PERSON] = 1
    MEDIUM_LAYOUT[7, 2, PERSON] = 1

    HARD_LAYOUT = np.zeros((15, 15, len(OBJECTS)), dtype=bool)
    HARD_LAYOUT[14, 7, ROBOT] = 1
    HARD_LAYOUT[0, 14, WATER] = 1
    HARD_LAYOUT[1, 0, PERSON] = 1
    HARD_LAYOUT[5, 5, PERSON] = 1
    HARD_LAYOUT[9, 10, PERSON] = 1
    HARD_LAYOUT[4, 3, PERSON] = 1
    HARD_LAYOUT[10, 12, PERSON] = 1
    HARD_LAYOUT[11, 6, PERSON] = 1
    HARD_LAYOUT[13, 4, PERSON] = 1



    DEFAULT_LAYOUTS = [[(2, 4), (4, 1), (0, 1), (0, 0), (3, 1)], [(1, 3), (4, 3), (4, 0), (3, 3), (1, 4)],
                       [(3, 1), (2, 4), (4, 0), (2, 2), (2, 1)], [(1, 0), (0, 1), (0, 0), (1, 4), (1, 3)],
                       [(3, 1), (4, 0), (3, 4), (2, 2), (4, 3)], [(1, 2), (1, 4), (4, 1), (0, 4), (3, 3)],
                       [(1, 3), (4, 4), (0, 0), (0, 2), (3, 0)], [(1, 0), (4, 4), (2, 1), (0, 3), (3, 4)],
                       [(1, 4), (4, 4), (1, 3), (1, 2), (4, 3)], [(1, 3), (3, 1), (2, 3), (4, 4), (0, 0)]]
    MEDIUM_LAYOUTS = [[(0, 3), (5, 4), (0, 1), (5, 2), (5, 1), (3, 0)], [(3, 3), (0, 3), (3, 4), (4, 3), (2, 3), (4, 1)],
         [(0, 3), (1, 0), (3, 3), (1, 5), (0, 1), (4, 3)], [(4, 1), (2, 0), (0, 3), (5, 2), (3, 5), (5, 4)],
         [(2, 2), (0, 4), (4, 2), (3, 1), (1, 5), (1, 1)], [(5, 4), (4, 5), (3, 1), (1, 4), (2, 1), (4, 2)],
         [(3, 5), (0, 1), (5, 5), (1, 5), (2, 4), (1, 4)], [(5, 5), (1, 4), (1, 0), (2, 0), (5, 0), (0, 0)],
         [(3, 1), (1, 3), (2, 3), (5, 0), (5, 4), (2, 2)], [(1, 1), (0, 0), (4, 4), (3, 4), (3, 0), (1, 2)]]
    HARD_LAYOUTS = [[(2, 3), (2, 5), (1, 1), (6, 1), (3, 6), (3, 2), (2, 4)], [(4, 6), (5, 1), (6, 6), (6, 5), (4, 0), (5, 0), (4, 3)],
         [(2, 2), (6, 6), (6, 0), (1, 5), (4, 4), (0, 4), (5, 3)], [(4, 5), (2, 4), (0, 5), (4, 1), (4, 6), (4, 0), (1, 1)],
         [(6, 2), (1, 1), (2, 6), (2, 0), (4, 1), (1, 0), (3, 2)], [(3, 6), (6, 3), (3, 2), (4, 4), (6, 4), (0, 2), (6, 5)],
         [(5, 1), (3, 2), (1, 0), (6, 1), (0, 5), (5, 5), (0, 6)], [(6, 2), (4, 0), (0, 5), (5, 3), (3, 1), (0, 1), (1, 0)],
         [(5, 1), (4, 1), (2, 3), (1, 6), (6, 1), (3, 0), (1, 2)], [(2, 1), (3, 5), (6, 1), (1, 4), (0, 5), (6, 2), (0, 4)]]

    # Actions
    ACTIONS = UP, DOWN, LEFT, RIGHT = range(4)

    # Reward for quenching
    QUENCH_REWARD = 0.1
    WATER_PICKUP_REWARD = 0.01
    MAX_REWARD = max(QUENCH_REWARD, WATER_PICKUP_REWARD)

    # For rendering
    TOKEN_IMAGES = {
        ROBOT : plt.imread(get_asset_path('robot.png')),
        ROBOT_WITH_WATER : plt.imread(get_asset_path('robot_with_water.png')),
        WATER : plt.imread(get_asset_path('water.png')),
        PERSON : plt.imread(get_asset_path('person.png')),
        QUENCHED_PERSON : plt.imread(get_asset_path('quenched_person.png')),
    }

    OBJECT_CHARS = {
        ROBOT : "R",
        ROBOT_WITH_WATER : "R",
        WATER : "W",
        PERSON : "P",
        QUENCHED_PERSON : "X",
    }

    def __init__(self, layout=None, mode='default', num_alias_per_action=1):
        if layout is None:
            if mode == 'default':
                layout = self.DEFAULT_LAYOUT
            elif mode == 'medium':
                layout = self.MEDIUM_LAYOUT
            elif mode == 'hard':
                layout = self.HARD_LAYOUT
            else:
                raise Exception("Unrecognized mode.")

        self._initial_layout = layout
        self._layout = layout.copy()
        self._num_alias_per_action = num_alias_per_action

        self._mode = mode
        self.problems = range(10)

    def fix_problem_index(self, index):
        original_layout = self._layout.copy()

        if self._mode == 'default':
            raw = self.DEFAULT_LAYOUTS[index]
        elif self._mode == 'medium':
            raw = self.MEDIUM_LAYOUTS[index]
        elif self._mode == 'hard':
            raw = self.HARD_LAYOUTS[index]
        else:
            raise Exception("Unrecognized mode.")

        ## create layout based on list of random generated positions
        shape = len(raw)
        layout = np.zeros((shape, shape, len(self.OBJECTS)), dtype=bool)
        layout[raw[0][0], raw[0][1], self.ROBOT] = 1
        layout[raw[1][0], raw[1][1], self.WATER] = 1
        for i in range(2, len(raw)):
            layout[raw[i][0], raw[i][1], self.PERSON] = 1

        self._initial_layout = layout
        self._layout = layout.copy()

    def reset(self):
        self._layout = self._initial_layout.copy()
        return self.get_state(), {}

    def get_all_actions(self):
        actions = [a for a in self.ACTIONS]
        num_actions = len(actions)
        assert actions[0] == 0 and actions[-1] == len(actions) - 1
        for _ in range(self._num_alias_per_action):
            for _ in range(num_actions):
                actions.append(max(actions)+1)
        return actions

    def get_possible_actions(self):
        return self.ACTIONS

    def step(self, action):
        # Handle action aliasing
        action = action % len(self.ACTIONS)

        # Start out reward at 0
        reward = 0

        # Move the robot
        rob_r, rob_c = None, None
        for robot_type in [self.ROBOT, self.ROBOT_WITH_WATER]:
            where_robot = np.argwhere(self._layout[..., robot_type])
            if len(where_robot) == 0:
                continue
            assert len(where_robot) == 1 or rob_r is not None, "Multiple robots in grid"
            rob_r, rob_c = where_robot[0]
            dr, dc = {self.UP : (-1, 0), self.DOWN : (1, 0), 
                      self.LEFT : (0, -1), self.RIGHT : (0, 1)}[action]
            new_r, new_c = rob_r + dr, rob_c + dc
            if 0 <= new_r < self._layout.shape[0] and 0 <= new_c < self._layout.shape[1]:
                # Remove old robot
                self._layout[rob_r, rob_c, robot_type] = 0
                # Add new robot
                self._layout[new_r, new_c, robot_type] = 1
                # Update local vars
                rob_r, rob_c = new_r, new_c
        assert rob_r is not None, "Missing robot in grid"

        # Handle water pickup
        if self._layout[rob_r, rob_c, self.ROBOT] and self._layout[rob_r, rob_c, self.WATER]:
            # Make robot have water
            self._layout[rob_r, rob_c, self.ROBOT_WITH_WATER] = 1
            self._layout[rob_r, rob_c, self.ROBOT] = 0
            # Remove water from grid
            self._layout[rob_r, rob_c, self.WATER] = 0
            # Reward for water pickup
            reward += self.WATER_PICKUP_REWARD

        # Handle people quenching
        if self._layout[rob_r, rob_c, self.ROBOT_WITH_WATER] and self._layout[rob_r, rob_c, self.PERSON]:
            # Quench person
            self._layout[rob_r, rob_c, self.QUENCHED_PERSON] = 1
            self._layout[rob_r, rob_c, self.PERSON] = 0
            # Reward for quenching
            reward += self.QUENCH_REWARD

        # Check done: all people quenched
        done = (len(np.argwhere(self._layout[..., self.PERSON])) == 0)

        return self.get_state(), reward, done, {}

    def render(self, dpi=150):
        return render_from_layout(self._layout, self._get_token_images, dpi=dpi)

    def _get_token_images(self, obs_cell):
        images = []
        for token in [self.ROBOT, self.ROBOT_WITH_WATER,self. WATER, 
                      self.PERSON, self.QUENCHED_PERSON]:
            if obs_cell[token]:
                images.append(self.TOKEN_IMAGES[token])
        return images

    def state_to_str(self, state):
        layout = np.full(self._initial_layout.shape[:2], "O", dtype=object)
        for i, j, k in state:
            layout[i, j] = self.OBJECT_CHARS[k]
        return '\n' + '\n'.join(''.join(row) for row in layout)

    def get_state(self):
        return tuple(sorted(map(tuple, np.argwhere(self._layout))))

    def set_state(self, state):
        self._layout = np.zeros_like(self._initial_layout)
        for i, j, k in state:
            self._layout[i, j, k] = 1

    @functools.lru_cache(maxsize=1000)
    def compute_reward(self, state, action):
        original_state = self.get_state()
        self.set_state(state)
        _, reward, _, _ = self.step(action)
        self.set_state(original_state)
        return reward

    @functools.lru_cache(maxsize=1000)
    def compute_transition(self, state, action):
        original_state = self.get_state()
        self.set_state(state)
        next_state, _, _, _ = self.step(action)
        self.set_state(original_state)
        return next_state

    def get_successor_state(self, state, action):
        return self.compute_transition(state, action)

    @functools.lru_cache(maxsize=1000)
    def compute_done(self, state, action):
        original_state = self.get_state()
        self.set_state(state)
        _, _, done, _ = self.step(action)
        self.set_state(original_state)
        return done

    @functools.lru_cache(maxsize=1000)
    def check_goal(self, state):
        for i, j, k in state:
            if k == self.PERSON:
                return False
        return True

def test_env():
    import imageio

    max_num_steps = 10
    use_default_layout = False

    if use_default_layout:
        layout = None
        dpi = 50
    else:
        layout = np.zeros((15, 15, len(WaterDeliveryEnv.OBJECTS)), dtype=bool)
        layout[14, 7, WaterDeliveryEnv.ROBOT] = 1
        layout[0, 14, WaterDeliveryEnv.WATER] = 1
        layout[1, 0, WaterDeliveryEnv.PERSON] = 1
        layout[5, 5, WaterDeliveryEnv.PERSON] = 1
        layout[9, 10, WaterDeliveryEnv.PERSON] = 1
        layout[4, 3, WaterDeliveryEnv.PERSON] = 1
        layout[10, 12, WaterDeliveryEnv.PERSON] = 1
        layout[11, 6, WaterDeliveryEnv.PERSON] = 1
        layout[13, 4, WaterDeliveryEnv.PERSON] = 1
        dpi = 150

    images = []
    env = WaterDeliveryEnv(layout)
    state, _ = env.reset()
    images.append(env.render(dpi=dpi))
    print("Initial state:", state)
    for _ in range(max_num_steps):
        action = np.random.choice(env.ACTIONS)
        print("Taking action", action)
        state, reward, done, _ = env.step(action)
        print("State:", state)
        print("Reward, Done:", reward, done)
        images.append(env.render(dpi=dpi))
        if done:
            break
    outfile = "water_delivery_random_actions.mp4"
    imageio.mimsave(outfile, images)
    print("Wrote out to", outfile)

def generate_layouts():

    import random

    shapes = {'default': 5, 'medium': 6, 'hard': 7}
    for mode in shapes.keys():
        shape = shapes[mode]
        raws = []
        for i in range(10):
            positions = []
            while len(positions) < shape:
                x = random.randint(0, shape-1)
                y = random.randint(0, shape-1)
                if (x, y) not in positions:
                    positions.append((x, y))
            raws.append(positions)
        print(raws)


if __name__ == "__main__":

    # test_env()
    generate_layouts()



