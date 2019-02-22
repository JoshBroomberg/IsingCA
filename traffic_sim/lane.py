import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from ..framework import CASim

DEBUG = False


class Lane(CASim):
    """
    Simulation of a single lane of traffic.

    This class simulates a single lane, with a variable initial density of cars.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the sim by calling the parent constructor.

        No custom config is needed for this simulation. All state and config
        parameters are set in a standardized format by the CASim class.

        The Multilane Highway is composed of a number of individual lane
        simulations. So this class has minor modification to support individual
        or combined usage.
        """
        super().__init__(*args, **kwargs)

    def _setup(self):
        # 1 where car, zero elsewhere.
        self.current_state = np.full(self.lane_length, -1)

        # Add cars with initial velocity of zero, achieved through addition.
        cars = (np.random.random(self.lane_length) < self.traffic_density)

        self.current_state += cars*(1+np.random.randint(
            self.max_velocity, size=self.lane_length))

        # Ensure there is at least one car, even at low densities.
        if np.sum(cars) == 0:
            self.current_state[np.random.randint(self.lane_length)] = 0

    def step(self):
        """
        Execute one step of the simulation.

        This overrides the parent method to skip the observation, which
        is done during the update step to collect data on velocity prior
        to position update.
        """
        self.steps += 1
        self._update()

    def _update(self):
        """
        Main update function, advances all cars according to the simulation
        rules.

        Note, the update procedure is as follows:
        - 1. If the car is below v_max, increase speed by one
        regardless of space available (the check for space is skipped.)
        - 2. Clip speed down to the max allowed by the distance to the next
        car.
        - 3. (Optional): execute custom good/bad driver behavior
        - 4. Update position.
        """

        # Always start next state at current state, this will help us below.
        self.next_state = self.current_state.copy()

        # Accelerate all cars by 1.
        self._accelerate()

        # If enabled, adjust speed to maintain an equal distance
        # between front and back cars.
        if self.maintain_braking_distance:
            self._maintain_braking_distance()

        # Reduce speed to avoid collisions (clip to gap).
        self._avoid_collisions()

        # Randomly slow down, sometimes.
        self._random_slow_down()

        # We are now after velocity update, prior to position update.
        # Record an observation.
        self._observe(self.next_state)

        # Move cars based on new velocity
        car_indeces = self._get_car_locations()
        car_velocities = self.next_state[car_indeces]
        new_car_indeces = car_indeces + car_velocities  # new car positions

        # place cars at new positions with same velocity, using periodic bound.
        self.next_state = np.full(self.lane_length, -1)  # 'empty' array
        self.next_state[new_car_indeces % self.lane_length] = car_velocities

        self.current_state = self.next_state

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
        """
        Implements smart driving acceleration, which aims to keep the driver
        in the middle of the car in front and behind.
        """
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

    # MULTILANE Functions
    def swap_into_lane(self, other_lane):
        """
        Evaluate given other_lane to determine if cars should swap lanes based
        on rules in Rickert, Nagel, Schreckenberg, Latourd.

        This function is used by the Highway class to perform lane changes.
        When called, the cars in this lane will swap to the other lane if
        appropriate.
        """
        current_lane_locations = self._get_car_locations()

        # Return if there are no cars in this lane.
        if len(current_lane_locations) == 0:
            return

        # Find current and desired velocity. Desired velocity is the threshold
        # at which cars choose to leave current lane (if unachievable) and to
        # swith to the other lane (if achievable).
        current_lane_velocity = self.current_state[current_lane_locations]
        desired_velocity = current_lane_velocity + 1

        # Find the gaps to the car in front and behind the position of all
        # cars in this lane if they were to switch to the other lane.
        gap_to_back_car_in_other_lane, gap_to_front_car_in_other_lane = \
            self._find_gaps_in_other_lane(other_lane)

        # Find the gaps between cars in this lane.
        gap_to_front_car_in_current_lane = self._find_gaps(full_dim=False)

        # DECISION LOGIC: the rules from the paper are implement below.

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

        # EXECUTE SWAP
        swap_locations = current_lane_locations[should_swap] # locations to swap.
        other_lane.current_state[swap_locations] = self.current_state[swap_locations]
        self.current_state[swap_locations] = -1

        # Return number of swaps.
        return len(swap_locations)

    # UTILITY SUBFUNCTIONS
    def _get_car_locations(self):
        car_present = self.current_state > -1
        locations = np.where(car_present)[0]
        return locations

    def _find_gaps(self, gap_type="FRONT", full_dim=True):
        """
        Find the disance from each car to the next car, where the meaning of
        next is controlled by gap_type.

        If gap_type is FRONT: returns the gap to the car in front, with zeros
        in all locations which don't have cars.

        If gap_type is BACK: returns the gap to the car behind, with zeros
        in all locations which don't have cars.

        RETURNS: if full_dim, an array with the gap to the car in front/behind
        in the position of each car. If not full_dim, returns an ordered
        list of gaps for the cars in the lane.
        """

        # Get the location of all cars.
        car_locations = self._get_car_locations()

        if len(car_locations) != 0:
            # Add first car location to end of list for periodic bound.
            car_locations = np.append(car_locations, car_locations[0])

        # Get gaps between cars, mod lane length.
        gaps_to_next_car = (np.ediff1d(car_locations) - 1) % self.lane_length

        # Front/back conversion
        if gap_type == "BACK":
            # The gap to the car behind is the forward gap from the behind
            # car, so roll forward one.
            gaps_to_next_car = np.roll(gaps_to_next_car, 1)
        elif gap_type != "FRONT":
            raise Exception("Invalid gap type, must be one of: [front, back].")

        if full_dim:
            # Move the spaces between cars back into full lane dimensionality
            full_gaps_to_next_car = np.zeros(self.lane_length, dtype=int)
            full_gaps_to_next_car[car_locations[:-1]] = gaps_to_next_car
            return full_gaps_to_next_car
        else:
            return gaps_to_next_car

    def _find_gaps_in_other_lane(self, other_lane):
        """
        Finds the gaps between the cars in the this lane and the car
        in front and behind if they were to swap to the other lane.

        RETURNS: A tuple of arrays where the first has the back gap and the
        second the front gap for each car. Both arrays are in full dimensionality.
        """
        current_lane_locations = self._get_car_locations()
        new_lane_locations = other_lane._get_car_locations()

        # If there are no cars in the other lane, return an array of
        # values at the max length possible (indicating maximum freedom).
        if len(new_lane_locations) == 0:
            max_arr = np.full(len(current_lane_locations), self.lane_length)
            return max_arr, max_arr

        rng = np.arange(len(new_lane_locations))

        # Find the position of the car behind the insert location.
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

    def draw(self, title, ax=None):
        """
        Display a static representing of the sim.

        This static representing uses the notation of Nagel and Schreckenberg.
        """
        if ax is None:
            ax = plt.axes()

        ax.axis('off')
        ax.title.set_text(title)

        ax.imshow(self.history, vmin=-1, vmax=0,
               cmap=cm.get_cmap('binary', 2))
