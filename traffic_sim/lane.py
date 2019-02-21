import numpy as np
import matplotlib.pyplot as plt

from ..framework import CASim

DEBUG = False
class Lane(CASim):
    """HELPFUL DESCRIPTION"""

    def __init__(self, *args, **kwargs):
        self.gap_cache = {}
        self.location_cache = {}

        super().__init__(*args, **kwargs)

    def _setup(self):
        # 1 where car, zero elsewhere.
        self.current_state = np.full(self.lane_length, -1)

        # Add cars with random initial velocity
        cars = (np.random.random(self.lane_length) < self.traffic_density)
        self.current_state += cars

        # Ensure there is at least one car!
        if np.sum(cars) == 0:
            self.current_state[0] = 0

    def draw(self, step_to_draw=None, prepend=""):
        """Display the sim."""
        def draw_state(state): return print(prepend + ''.join('.' if x == self.STATES["EMPTY"] else str(
            x) for x in state), end="")

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
        car_velocities = self.next_state[car_indeces]
        new_car_indeces = car_indeces + car_velocities  # new car positions

        # place cars at new positions with same velocity, using periodic bound.
        self.next_state = np.full(self.lane_length, -1)  # 'empty' array
        self.next_state[new_car_indeces % self.lane_length] = car_velocities

        self.current_state = self.next_state

    def swap_into_lane(self, other_lane):
        current_lane_locations = self._get_car_locations()

        if len(current_lane_locations) == 0:
            return

        current_lane_velocity = self.current_state[current_lane_locations]
        desired_velocity = current_lane_velocity + 1

        gap_to_back_car_in_other_lane, gap_to_front_car_in_other_lane = \
            self._find_gaps_in_other_lane(other_lane)

        gap_to_front_car_in_current_lane = self._find_gaps(full_dim=False)

        # VALIDATE
        if DEBUG:
            other_front_locs = (current_lane_locations
                          + gap_to_front_car_in_other_lane + 1) % self.lane_length
            other_back_locs = (current_lane_locations
                          - gap_to_back_car_in_other_lane - 1) % self.lane_length
            if -1 in other_lane.current_state[other_front_locs]:
                raise Exception("Front gap calculation error!")

            if -1 in other_lane.current_state[other_back_locs]:
                raise Exception("Back gap calculation error!")
        # END VALIDATE

        current_lane_disadvantageous = (gap_to_front_car_in_current_lane
                                        < desired_velocity)

        other_lane_advantagous = (gap_to_front_car_in_other_lane
                                  > desired_velocity)

        other_lane_available = (gap_to_back_car_in_other_lane
                                > self.other_lane_back_gap_swap_threshold)

        chooses_swap = (np.random.random(len(other_lane_available))
                        < self.p_lane_change)

        should_swap = (current_lane_disadvantageous
                       * other_lane_advantagous
                       * other_lane_available
                       * chooses_swap)

        swap_locations = current_lane_locations[should_swap]

        other_lane.current_state[swap_locations] = self.current_state[swap_locations]
        self.current_state[swap_locations] = -1

    # UTILITY SUBFUNCTIONS
    def _get_car_locations(self):
        # Memoize function to avoid unnecessary recalutation.
        # loc_cache_key = str(self.current_state)
        # if loc_cache_key in self.location_cache:
        #     return self.location_cache[loc_cache_key]

        car_present = self.current_state > -1
        locations = np.where(car_present)[0]

        # self.location_cache[loc_cache_key] = locations
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
        # gap_cache_key = (
        #     self.steps,
        #     gap_type,
        #     full_dim)
        # if gap_cache_key in self.gap_cache:
        #     return self.gap_cache[gap_cache_key]

        # Get the location of all cars.
        car_locations = self._get_car_locations()

        if len(car_locations) != 0:
            # Add first car location to end of list for periodic bound.
            car_locations = np.append(car_locations, car_locations[0])

        # Get gaps between cars, mod lane length.
        gaps_to_next_car = (np.ediff1d(car_locations) - 1) % self.lane_length

        # Front/back conversion
        if gap_type == "BACK":
            gaps_to_next_car = np.roll(gaps_to_next_car, 1)
        elif gap_type != "FRONT":
            raise Exception("Invalid gap type, must be one of: [front, back].")

        if full_dim:
            # Move the spaces between cars back into full sim dimensionality
            full_gaps_to_next_car = np.zeros(self.lane_length, dtype=int)
            full_gaps_to_next_car[car_locations[:-1]] = gaps_to_next_car
            # self.gap_cache[gap_cache_key] = full_gaps_to_next_car
            return full_gaps_to_next_car
        else:
            # self.gap_cache[gap_cache_key] = gaps_to_next_car
            return gaps_to_next_car

    def _find_gaps_in_other_lane(self, other_lane):
        current_lane_locations = self._get_car_locations()
        new_lane_locations = other_lane._get_car_locations()

        if len(new_lane_locations) == 0:
            max_arr = np.full(len(current_lane_locations), self.lane_length)
            return max_arr, max_arr

        rng = np.arange(len(new_lane_locations))

        # Find the position of the car behind insert location.
        new_lane_min_insert_index = np.searchsorted(
            new_lane_locations,
            current_lane_locations,
            side="right",
            sorter=rng)

        back_car_location = new_lane_locations[new_lane_min_insert_index-1]
        gap_to_back_car_in_other_lane = (
            (current_lane_locations - back_car_location) % self.lane_length)-1

        new_lane_max_insert_index = np.searchsorted(
            new_lane_locations,
            current_lane_locations,
            side="left",
            sorter=rng)

        front_car_location = new_lane_locations[new_lane_max_insert_index % len(
            new_lane_locations)]
        gap_to_front_car_in_other_lane = (
            (front_car_location - current_lane_locations) % self.lane_length)-1

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
        gap_to_front_car = self._find_gaps(gap_type="FRONT")

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
        should_decrease = (np.random.random(
            self.lane_length) < self.p_slow_down)
        can_decrease = (self.next_state > 0)

        self.next_state -= can_decrease*should_decrease
