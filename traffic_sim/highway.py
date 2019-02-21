from ..framework import CASim
from . import Lane
import numpy as np


class Highway(CASim):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def _setup(self):
        self.lanes = np.array([Lane(*self.args, **self.kwargs)
                      for _ in range(self.num_lanes)])

    def step(self):
        """Execute one step of the simulation. Override to remove observation."""
        self.steps += 1
        self._update()

    def _update(self):
        lane_swap_deltas = [-1, 1] if self.num_lanes > 2 else [1]
        for lane_index, lane in enumerate(self.lanes):
            for swap_delta in lane_swap_deltas:
                other_lane_index = (lane_index+swap_delta)%self.num_lanes
                other_lane = self.lanes[other_lane_index]

                lane.swap_into_lane(other_lane)

        # Run updates on all lanes
        for lane in self.lanes:
            lane.step()

    def draw(self):
        for step in range(self.steps):
            for lane_num, lane in enumerate(self.lanes):
                lane.draw(step_to_draw=step, prepend=f"L{lane_num + 1}: ")
                print()
            print("\n")

    def _observe(self)

    def video(self):
        print(self.history)
