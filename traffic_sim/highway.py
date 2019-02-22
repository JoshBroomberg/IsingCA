from ..framework import CASim
from . import Lane
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Highway(CASim):
    """
    This class implements a Multilane road. It is itself a CASim, conforming
    to the API so that it can be used with CAExperiment. It is composed of
    N single-lane simulations and orchestrates interactions between them.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the highway. All setup is done by CASim. The args and kwargs
        are saved for Lane construction.
        """
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def _setup(self):
        # Build num_lanes Lane objects.
        self.lanes = np.array([Lane(*self.args, **self.kwargs)
                               for _ in range(self.num_lanes)])

        # The state of the Highway is the state of each of the lanes in an
        # num_lanes X lane_length array.
        self.current_state = np.full(self.dim, -1)

    def _update(self):
        """
        A highway update consists of lane changes, followed by forward
        movement in each lane.
        """
        self._execute_lane_changes()
        self._execute_forward_movement()

    def _execute_forward_movement(self):
        # Run standard updates on all lanes and update the state of the
        # highway.
        for lane_index, lane in enumerate(self.lanes):
            lane.step()
            self.current_state[lane_index, :] = lane.history[-1]

    def _execute_lane_changes(self):
        # Decide which lanes will be swapped. If there are more than 2
        # lanes, each lane can swap with the next and prior lane. If there
        # are two, then only the 'next' lane (assuming periodic lane access)
        # and if one, then no lanes.
        if self.num_lanes > 2:
            lane_swap_deltas = [-1, 1]
        elif self.num_lanes == 2:
            lane_swap_deltas = [1]
        else:
            lane_swap_deltas = []

        # If there are swaps to execute, execute them for each lane and each
        # destination.
        if len(lane_swap_deltas) > 0:
            for lane_index, lane in enumerate(self.lanes):
                for swap_delta in lane_swap_deltas:
                    # Periodic lane access, ensures each lane swaps with
                    # the same number of lanes.
                    other_lane_index = (lane_index+swap_delta) % self.num_lanes
                    other_lane = self.lanes[other_lane_index]
                    lane.swap_into_lane(other_lane)

    def draw(self):
        """
        Represent each lane in the static output.
        """
        for step in range(self.steps):
            for lane_num, lane in enumerate(self.lanes):
                lane.draw(step_to_draw=step, prepend=f"L{lane_num + 1}: ")
                print()
            print("\n")
