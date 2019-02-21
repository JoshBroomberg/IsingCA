from collections import defaultdict

import numpy as np

from matplotlib import animation
import matplotlib.pyplot as plt


class CASim():
    """
    Framework to build, run and analyze Cellular Automata Simulation.
    """

    COLOR_WHITE = [255, 255, 255]
    COLOR_GREEN = [0, 255, 0]
    COLOR_YELLOW = [255, 255, 0]
    COLOR_ORANGE = [255, 165, 0]
    COLOR_RED = [255, 0, 0]
    COLOR_BLUE = [0, 0, 255]
    COLOR_BLACK = [0, 0, 0]
    COLOR_GREY = [220, 220, 220]

    MAX_STEPS = 10**3

    def __init__(self, dim, neighborhood_radii, sim_params, sim_states,
                 metrics={}, start_stats={}, end_stats={}, *args, **kwargs):

        # Config
        self.dim = dim
        self.neighborhood_radii = neighborhood_radii

        # Set params and define states + colors.
        self._configure(sim_params, sim_states)

        # State
        self.current_state = np.zeros(self.dim)
        self.next_state = np.zeros(self.dim)
        self._setup()

        # History and metrics
        self.steps = 0
        self.history = []
        self.metrics = metrics
        self.start_stats = start_stats
        self.end_stats = end_stats

        self.METRIC_RESULTS = defaultdict(list)
        self.START_STATS_RESULTS = defaultdict(int)
        self.END_STATS_RESULTS = defaultdict(int)

        self._initial_observation()  # baseline observation

    # PROPERTIES
    @property
    def neighborhood_size(self):
        return np.prod([(self.neighborhood_radii[i]*2 + 1)**2 for i in range(2)])

    @property
    def sim_size(self):
        return self.dim[0]*self.dim[1]

    # CUSTOM METHODS
    def _setup(self):
        raise NotImplementedError

    def _update(self):
        raise NotImplementedError

    # BASE METHODS
    def _configure(self, sim_params, sim_states):
        # Set parameters
        for param_name, param_value in sim_params.items():
            setattr(self.__class__, param_name, param_value)

        # Setup states
        self.STATES = {}  # name - number amp
        self.STATE_COLORS = {}  # name - color map
        for state_name, (state_number, state_color) in sim_states.items():
            self.STATES[state_name] = state_number
            self.STATE_COLORS[state_name] = state_color

        num_states = len(self.STATES)
        self.palette = np.zeros((num_states, 3), dtype=int)
        for i, (state, color) in enumerate(self.STATE_COLORS.items()):
            self.palette[i, :] = color

    def _initial_observation(self):
        self.steps += 1
        for start_stat_name, start_stat_lambda in self.start_stats.items():
            self.START_STATS_RESULTS[start_stat_name] = start_stat_lambda(self)

        self._observe()

    def _observe(self, state=None):
        if state is None:
            state = self.current_state

        self.history.append(state.copy())

        for metric_name, metric_lambda in self.metrics.items():
            self.METRIC_RESULTS[metric_name].append(metric_lambda(self))

    def _final_observation(self):
        for end_stat_name, end_stat_lambda in self.end_stats.items():
            self.END_STATS_RESULTS[end_stat_name] = end_stat_lambda(self)

    def step(self):
        self.steps += 1
        self._update()
        self._observe()

    def run_until_stable(self, max_steps=None, early_stop=True):
        step_lim = max_steps or self.MAX_STEPS
        while self.steps < step_lim:
            self.step()

            if early_stop and (self.steps >= 3 and np.array_equal(
                    self.history[-1], self.history[-2])):
                break

        self._final_observation()

    # GENERAL UTILITY METHODS
    def get_random_cell_coord(self):
        return tuple([np.random.choice(range(self.dim[i])) for i in [0, 1]])

    # UPDATE UTILITY METHODS
    def _count_all_in_state_in_neighborhood(self, state=1):
        counts = np.zeros(self.dim)
        del_x_range = range(-self.neighborhood_radii[0],
                            self.neighborhood_radii[0]+1)
        del_y_range = range(-self.neighborhood_radii[1],
                            self.neighborhood_radii[1]+1)

        # Place 1s at all countable locations, 0s elsewhere.
        countable_state = (self.current_state == state).astype(int)
        for del_x in del_x_range:
            for del_y in del_y_range:
                countable_state = np.roll(countable_state, del_x, axis=1)
                countable_state = np.roll(countable_state, del_y, axis=0)
                counts += countable_state
                countable_state = (self.current_state == state).astype(int)

        return counts

    def _count_all_in_state(self, state):
        """
        Metric collection helper: count all cells in given state.
        """
        return np.sum((self.current_state == state).astype(int))

    def _density_in_state(self, state, *args):
        """
        Metric collection helper: count all cells in given state.
        """
        return self._count_all_in_state(state=state)/self.sim_size

    # DISPLAY
    def draw(self, step_to_draw=None):
        step_to_draw = step_to_draw or (self.steps - 1)
        plt.title("State at step {}".format(step_to_draw + 1))
        plt.imshow(self.palette[self.history[step_to_draw]])
        plt.show()

    def get_all_stats(self):
        return {**self.START_STATS_RESULTS, **self.END_STATS_RESULTS}

    def display_stats(self):
        if len(self.metrics) > 0:
            plt.figure(figsize=(12, 6))
            plt.title("Metrics at step {}".format(self.steps))
            for metric_name, metric_lambda in self.metrics.items():
                plt.plot(range(self.steps), self.METRIC_RESULTS[metric_name],
                         label=metric_name)
            plt.legend()
            plt.show()

        if len(self.get_all_stats()) > 0:
            print("======================================")
            print("{:25}| {:8} |".format("stat", "value"))
            print("______________________________________")
            for stat, stat_value in self.get_all_stats().items():
                print("{:25}| {:8.2f} |".format(stat, stat_value))

            print("======================================\n")

    def video(self):
        fig = plt.figure(figsize=(12, 4))
        plt.title("Animation")
        plt.grid(None)
        plt.axis("off")

        # ims is a list of lists, each row is a list of artists to draw in the
        # current frame; here we are just animating one artist, the image, in
        # each frame
        ims = []
        for state in self.history:
            if len(state.shape) == 1:
                state = np.array([state])
            im = plt.imshow(self.palette[state + 1], animated=True)
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,
                                        repeat_delay=10000)

        plt.close()
        return ani
