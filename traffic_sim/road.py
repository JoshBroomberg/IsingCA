import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

from ..framework import CASim


class Road(CASim):
    """HELPFUL DESCRIPTION"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gap_cache = {}

    def _setup(self):
        # 1 where car, zero elsewhere.
        self.current_state = np.full(self.dim, -1)

        # Add cars with random initial velocity
        cars = (np.random.random(self.dim) < self.traffic_density)
        self.current_state += cars*(1 + np.random.randint(
            self.max_velocity, size=self.dim))

        # Ensure there is at least one car!
        if np.sum(cars) == 0:
            self.current_state[0, 1] = np.random.randint(
                1, self.max_velocity+1)

    def draw(self, step_to_draw=None):
        """Display the sim."""
        for state in self.history[1:]:
            print(''.join('.' if x == self.STATES["EMPTY"] else str(
                x) for x in state[0, :]))

    def step(self):
        """Execute one step of the simulation."""
        self.steps += 1
        self._update()

    def _update(self):
        # Note, the update is simplified as follows:
        # if the car is below v_max, always
        # increase speed by one regardless of space,
        # then clip speed down to the max allowed by
        # the distance. We do an unnecessary +1 in some cases, but this is
        # much more efficient than checking for validity twice.

        # Always start next state at current state, this will help us below.
        self.next_state = self.current_state.copy()

        # Accelerate all cars by 1.
        self._accelerate()

        # If enabled, adjust speed to maintain an equal distance
        # between front and back cars.
        if self.maintain_braking_distance:
            self._maintain_braking_distance()

        # Reduce speed to avoid collisions.
        self._avoid_collisions()

        # Randomly slow down.
        self._random_slow_down()

        # Observe after velocity update, prior to position update.
        self._observe(self.next_state)

        # Move cars based on new velocity
        car_indeces = np.where(self.next_state > -1)[1]
        car_velocities = self.next_state[0, car_indeces]
        new_car_indeces = car_indeces + car_velocities  # new car positions

        # place cars at new positions with same velocity, using periodic bound.
        self.next_state = np.full(self.dim, -1, dtype=int)  # 'empty' array
        self.next_state[0, new_car_indeces % self.dim[1]] = car_velocities

        self.current_state = self.next_state

    # UTILITY SUBFUNCTIONS
    def _find_gaps(self, gap_type="FRONT"):
        """
        Find the disance from each car to the next car, where the meaning of next is controller by gap_type.

        If gap_type is FRONT: returns the gap to the car in front, with zeros
        in all locations which don't have cars.

        If gap_type is BACK: returns the gap to the car behind, with zeros
        in all locations which don't have cars.
        """

        # Memoize function to avoid unnecessary recalutation.
        gap_cache_key = (self.steps, gap_type)
        if gap_cache_key in self.gap_cache:
            return self.gap_cache[gap_cache_key]

        car_present = self.current_state > -1

        # Get the location of all cars.
        car_locations = np.where(car_present)[1]

        # Add first car location to end of list for periodic bound.
        car_locations = np.append(car_locations, car_locations[0])

        # Get gaps between cars, mod lane length.
        gaps_to_next_car = (np.ediff1d(car_locations) - 1) % self.dim[1]

        # Front/back conversion
        if gap_type == "BACK":
            gaps_to_next_car = np.roll(gaps_to_next_car, 1)
        elif gap_type != "FRONT":
            raise Exception("Invalid gap type, must be one of: [front, back].")

        # Move the spaces between cars back into full sim dimensionality
        full_gaps_to_next_car = np.zeros(self.dim, dtype=int)
        full_gaps_to_next_car[0, car_locations[:-1]] = gaps_to_next_car

        self.gap_cache[gap_cache_key] = full_gaps_to_next_car
        return full_gaps_to_next_car

    # UPDATE SUBFUNCTIONS
    def _accelerate(self):
        # increase velocity of all cars below max by 1.
        car_present = self.current_state > -1
        below_max = (self.current_state < self.max_velocity)
        self.next_state += car_present*below_max

    def _avoid_collisions(self):
        # Clip the speed of all cars to avoid collisions by choosing
        # the minimum of space and current velocity.
        gap_to_front_car = self._find_gaps()
        self.next_state = np.minimum(self.next_state, gap_to_front_car)

    def _maintain_braking_distance(self):
        # Find space to previous car by rolling the array of spaces to next
        # car forward one.
        gap_to_prev_car = self._find_gaps(gap_type="BACK")
        gap_to_front_car = self._find_gaps()

        # Find the target front gap as the midpoint of the distance
        # to the car in front and behind.
        spacing = np.vstack([gap_to_prev_car, gap_to_front_car])
        target_front_gap = np.mean(spacing, axis=0)

        # Apply acceleration or decceleration depending on if you are
        # too close/too far to the car in front. Use boolean arithmetic
        # for speed and clarity.

        accel = (target_front_gap < gap_to_front_car)
        decel = (target_front_gap > gap_to_front_car)

        can_accel = (self.next_state < self.max_velocity)
        can_decel = (self.next_state > 0)

        self.next_state += can_accel*accel
        self.next_state -= can_decel*decel

    def _random_slow_down(self):
        should_decrease = (np.random.random(self.dim) < self.p_slow_down)
        can_decrease = (self.next_state > 0)

        self.next_state -= can_decrease*should_decrease
