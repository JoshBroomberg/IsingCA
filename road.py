import numpy as np
from framework import CASim


class Road(CASim):
    """Wrapper class for multiple lanes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup(self):
        # 1 where car, zero elsewhere.
        self.current_state = np.full(self.dim, -1)

        # Add cars with random initial velocity
        cars = (np.random.random(self.dim) < self.traffic_density).astype(int)
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

    def _update(self):
        # Always start next state at current state, this will help us below.
        self.next_state = self.current_state.copy()

        # Note, the update is simplified as follows:
        # if the car is below v_max, always
        # increase speed by one regardless of space,
        # then clip speed down to the max allowed by
        # the distance. We do an unnecessary +1 in some cases, but this is
        # much more efficient than checking for validity twice.

        # Note 2: this numpy implementation is overkill for a 1-lane sim
        # where each lane can be processed in O(N) time. But it will be
        # faster for longer lanes and meaningful when there are more lanes
        # an the efficiency of numpy parallel array ops comes into play.

        car_present_bool = self.current_state > -1
        car_present = (car_present_bool).astype(int)

        # increase velocity of all cars below max by 1.
        below_max = (self.current_state < self.max_velocity).astype(int)
        self.next_state += car_present*below_max

        # Find the distances between each car using index as location
        car_indeces = np.where(car_present_bool)[1]
        spaces_to_next_car = np.ediff1d(car_indeces) - 1

        # Manually find the distance from the last car back
        # around to the first car to account for the periodic bounds.
        last_gap = (self.dim[1] - car_indeces[-1] - 1  # length of lane remaining
                    + car_indeces[0])  # distance to first car
        spaces_to_next_car = np.append(spaces_to_next_car, last_gap)

        # Move the spaces between cars back into full sim dimensionality
        full_spaces_to_next_car = np.zeros(self.dim, dtype=int)
        full_spaces_to_next_car[0, car_indeces] = spaces_to_next_car

        # Clip the speed of all cars to avoid collisions by choosing
        # the minimum of space and current velocity.
        spaces_available = np.maximum(0, full_spaces_to_next_car)
        self.next_state = np.minimum(self.next_state, spaces_available)

        # Smart driving step:
        if self.enable_buffer:
            # Find space to previous car by rolling the array of spaces to next
            # car forward one.
            spaces_to_prev_car = np.roll(spaces_to_next_car, 1)
            full_spaces_to_prev_car = np.zeros(self.dim, dtype=int)
            full_spaces_to_prev_car[0, car_indeces] = spaces_to_prev_car

            # Find the target distance as the midpoint/mean of the distance
            # to the car in front and behing.
            spacing = np.vstack(
                [full_spaces_to_prev_car, full_spaces_to_next_car])
            target_distance = np.mean(spacing, axis=0).astype(int)

            # Apply acceleration or decceleration depending on if you are
            # too close/too far to the car in front.
            accel = (target_distance < full_spaces_to_next_car).astype(int)
            decel = (target_distance > full_spaces_to_next_car).astype(int)
            can_decel = (self.next_state > 0).astype(int)

            self.next_state += accel
            self.next_state -= can_decel*decel

            # Reclip the speed to prevent collisions. This should never happen
            # given the code above but rather safe than sorry.
            self.next_state = np.minimum(self.next_state, spaces_available)

        # Randomization of velocity.
        should_decrease = (np.random.random(self.dim)
                           < self.p_slow_down).astype(int)
        can_decrease = (self.next_state > 0).astype(int)

        # remove 1 from random cars
        self.next_state -= can_decrease*should_decrease

        # Observe prior to motion update to recreate the paper's figures.
        self._observe(self.next_state)

        # Move cars based on velocity
        car_indeces = np.where(self.next_state > -1)[1]
        car_velocities = self.next_state[0, car_indeces]
        new_car_indeces = car_indeces + car_velocities  # new car positions

        # place cars at new positions with same velocity, periodic bound.
        self.next_state = np.full(self.dim, -1, dtype=int)  # 'empty' array
        self.next_state[0, new_car_indeces % self.dim[1]] = car_velocities

        self.current_state = self.next_state

    def step(self):
        """Execute one step of the simulation."""
        self.steps += 1
        self._update()
