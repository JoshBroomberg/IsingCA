import numpy as np
import matplotlib.pyplot as plt

from ..framework import CASim


class Lane(CASim):
    """HELPFUL DESCRIPTION"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gap_cache = {}
        self.location_cache = {}

    def _setup(self):
        # 1 where car, zero elsewhere.
        self.current_state = np.full(self.dim, -1)

        # Add cars with random initial velocity
        cars = (np.random.random(self.dim) < self.traffic_density)
        self.current_state += cars

        # Ensure there is at least one car!
        # TODO: Ensure this suffices across many lanes.
        if np.sum(cars) == 0:
            self.current_state[0, 1] = np.random.randint(
                1, self.max_velocity+1)

    def draw(self, step_to_draw=None, prepend=""):
        """Display the sim."""
        def draw_state(state): return print(prepend + ''.join('.' if x == self.STATES["EMPTY"] else str(
            x) for x in state[0, :]), end="")

        if step_to_draw is not None:
            draw_state(self.history[step_to_draw])
        else:
            for state in self.history[1:]:
                draw_state(state)
                print()

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
        car_indeces = self._get_car_locations()
        car_velocities = self.next_state[0, car_indeces]
        new_car_indeces = car_indeces + car_velocities  # new car positions

        # place cars at new positions with same velocity, using periodic bound.
        self.next_state = np.full(self.dim, -1, dtype=int)  # 'empty' array
        self.next_state[0, new_car_indeces % self.dim[1]] = car_velocities

        self.current_state = self.next_state

    def swap_into_lane(self, other_lane):
        current_lane_locations = self._get_car_locations()

        gap_to_back_car_in_other_lane, gap_to_front_car_in_other_lane = \
            self._find_gaps_in_other_lane(other_lane)

        gap_to_front_car_in_current_lane = self._find_gaps(full_dim=False)

        current_lane_disadvantageous = (gap_to_front_car_in_current_lane[0]
                                        < self.current_lane_front_gap_swap_tolerance)

        other_lane_advantagous = (gap_to_front_car_in_other_lane
                                  > self.other_lane_front_gap_swap_threshold)

        other_lane_available = (gap_to_back_car_in_other_lane
                                > self.other_lane_back_gap_swap_threshold)

        chooses_swap = (np.random.random((len(other_lane_available), ))
                        < self.p_lane_change)

        should_swap = (current_lane_disadvantageous
                       * other_lane_advantagous
                       * other_lane_available
                       * chooses_swap)

        swap_locations = current_lane_locations[should_swap]

        other_lane.current_state[0, swap_locations] = self.current_state[0, swap_locations]
        self.current_state[0, swap_locations] = -1

    # UTILITY SUBFUNCTIONS
    def _get_car_locations(self):
        # Memoize function to avoid unnecessary recalutation.
        if self.steps in self.location_cache:
            return self.location_cache[self.steps]

        car_present = self.current_state > -1
        locations = np.where(car_present)[1]

        self.location_cache[self.steps] = locations
        return locations

    def _find_gaps(self, gap_type="FRONT", full_dim=True):
        """
        Find the disance from each car to the next car, where the meaning of next is controller by gap_type.

        If gap_type is FRONT: returns the gap to the car in front, with zeros
        in all locations which don't have cars.

        If gap_type is BACK: returns the gap to the car behind, with zeros
        in all locations which don't have cars.
        """

        # Memoize function to avoid unnecessary recalutation.
        gap_cache_key = (self.steps, gap_type, full_dim)
        if gap_cache_key in self.gap_cache:
            return self.gap_cache[gap_cache_key]

        # Get the location of all cars.
        car_locations = self._get_car_locations()

        # Add first car location to end of list for periodic bound.
        car_locations = np.append(car_locations, car_locations[0])

        # Get gaps between cars, mod lane length.
        gaps_to_next_car = (np.ediff1d(car_locations) - 1) % self.dim[1]

        # Front/back conversion
        if gap_type == "BACK":
            gaps_to_next_car = np.roll(gaps_to_next_car, 1)
        elif gap_type != "FRONT":
            raise Exception("Invalid gap type, must be one of: [front, back].")

        if full_dim:
            # Move the spaces between cars back into full sim dimensionality
            full_gaps_to_next_car = np.zeros(self.dim, dtype=int)
            full_gaps_to_next_car[0, car_locations[:-1]] = gaps_to_next_car
            self.gap_cache[gap_cache_key] = full_gaps_to_next_car
            return full_gaps_to_next_car
        else:
            self.gap_cache[gap_cache_key] = gaps_to_next_car
            return gaps_to_next_car

    def _find_gaps_in_other_lane(self, other_lane):
        lane_length = self.dim[1]
        current_lane_locations = self._get_car_locations()
        new_lane_locations = other_lane._get_car_locations()

        rng = np.arange(len(new_lane_locations))

        # Find the position of the car behind insert location.
        new_lane_min_insert_index = np.searchsorted(
            new_lane_locations,
            current_lane_locations,
            side="right",
            sorter=rng)

        back_car_location = new_lane_locations[new_lane_min_insert_index-1]
        gap_to_back_car_in_other_lane = (
            (current_lane_locations - back_car_location) % lane_length)-1

        new_lane_max_insert_index = np.searchsorted(
            new_lane_locations,
            current_lane_locations,
            side="left",
            sorter=rng)

        front_car_location = new_lane_locations[new_lane_max_insert_index % len(
            new_lane_locations)]
        gap_to_front_car_in_other_lane = (
            (front_car_location - current_lane_locations) % lane_length)-1

        return gap_to_back_car_in_other_lane, gap_to_front_car_in_other_lane

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
