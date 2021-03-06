import numpy as np
import pandas as pd
import time as timer
from itertools import product
from multiprocessing import Pool


class CAExperiment():
    def __init__(self, num_repeats, sim_class, run_params,
                 sim_params, constant_sim_params, stat_names):

        self.num_repeats = num_repeats

        self.run_params = run_params  # params used at sim runtime

        self.sim_class = sim_class
        self.sim_params = sim_params  # params to experiment over
        self.constant_sim_params = constant_sim_params
        self.stat_names = stat_names

        # Validate params and set order
        self.ordered_params = []    # unique param ordering
        for param, param_val_list in self.sim_params.items():
            if not hasattr(self.sim_params, '__iter__'):
                raise Exception("All params must be specified in lists!")

            self.ordered_params.append(param)

        self._build_param_sets()

        num_trials = len(self.param_val_sets) * self.num_repeats
        num_columns = len(self.ordered_params) + len(self.stat_names)

        self.raw_results = np.empty((num_trials, num_columns), dtype=float)

        self.total_run_count = len(self.param_val_sets)*self.num_repeats

    def _build_param_sets(self):
        self.param_val_sets = []
        val_sets = product(*[self.sim_params[param]
                             for param in self.ordered_params])
        for val_set in val_sets:
            self.param_val_sets.append(list(val_set))

    def run_trial(self, args):
        run_index, param_index, param_val_set, param_set = args
        start_time = timer.time()

        sim_instance = self.sim_class(
                sim_params=param_set,
                **self.constant_sim_params)
        sim_instance.run_until_stable(**self.run_params)

        # Collect stats
        sim_stats = sim_instance.get_all_stats()
        stat_vals = [sim_stats[stat_name]
                     for stat_name in self.stat_names]

        # Store
        result_index = param_index*self.num_repeats + run_index
        result_row = param_val_set + stat_vals
        end_time = timer.time()
        if param_index + run_index == 0:
            iter_time = round(end_time - start_time, 3)
            print("Iteration time", iter_time)
            print("Predicted single core total time", round(
                self.total_run_count*iter_time, 3))

        return result_index, result_row

    def run(self, n_jobs=1, clear_func=None):

        # Build run args
        run_args = []
        for param_index, param_val_set in enumerate(self.param_val_sets):
            param_set = dict(zip(self.ordered_params, param_val_set))
            for run_index in range(self.num_repeats):
                run_args.append(
                    (run_index, param_index, param_val_set, param_set))

        with Pool(processes=n_jobs) as pool:
            job_count = 0
            for result_index, result_row in pool.imap_unordered(self.run_trial, run_args):
                self.raw_results[result_index, :] = result_row
                job_count += 1

                incomplete = len(run_args) - job_count

                if job_count % 25 == 0:
                    print("Percent done: {}".format(
                        100 - (100*incomplete/len(run_args))))
                    if clear_func:
                        timer.sleep(0.1)
                        clear_func()

        column_names = self.ordered_params + self.stat_names
        self.results = pd.DataFrame(self.raw_results, columns=column_names)
