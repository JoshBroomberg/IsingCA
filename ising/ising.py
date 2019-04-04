import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from ..framework import CASim

DEBUG = False

class Ising(CASim):
    """
    Simulation of the Ising model.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the sim by calling the parent constructor.

        No custom config is needed for this simulation. All state and config
        parameters are set in a standardized format by the CASim class.
        """
        super().__init__(*args, **kwargs)

    def _setup(self):
        # 1 where car, zero elsewhere.
        self.current_state = np.random.choice([-1, 1], size=self.dim)
        self.neighborhood_delta_indeces = np.array([
            [0, 0, 1, -1],
            [1, -1, 0, 0]
        ])

    def _get_neighbourhood_indeces(self, center):
        indeces = self.neighborhood_delta_indeces + \
            np.array(center).reshape((-1, 1))

        indeces[0, :] = indeces[0, :] % self.dim[0]
        indeces[1, :] = indeces[1, :] % self.dim[1]

        return indeces

    def _update(self):
        """
        Main update function, advances all cells according to the simulation
        rules.
        """
        # Always start next state at current state, this will help us below.
        self.next_state = self.current_state.copy()

        random_cell_index = self.get_random_cell_coord()
        random_cell = self.current_state[random_cell_index]

        neighbour_indeces = self._get_neighbourhood_indeces(random_cell_index)
        neighbours = self.current_state[list(neighbour_indeces)]

        energy = -1*random_cell*np.sum(neighbours)
        flip_p = min(1, np.exp((2*energy)/self.temperature))

        if np.random.random() < flip_p:
            self.next_state[random_cell_index] *= -1

        self.current_state = self.next_state


    def draw(self, title, step=-1, ax=None):
        """
        Display a static representing of the sim.

        This static representing uses the notation of Nagel and Schreckenberg.
        """
        if ax is None:
            ax = plt.axes()

        ax.axis('off')
        ax.title.set_text(title)

        step_data = self.history[step]
        step_data[step_data<0] = 0
        ax.imshow(self.palette[step_data], vmin=-1, vmax=0,
               cmap=cm.get_cmap('binary', 2))
