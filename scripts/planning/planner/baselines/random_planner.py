from planner.planner import Planner
from planner.utils import random_view, uniform_sampling
import numpy as np


class RandomPlanner(Planner):
    def __init__(self, cfg):
        super().__init__(cfg)
        print("initial ")
        self.num_candidates = cfg["num_candidates"]
        self.view_change = cfg["view_change"]
        self.planning_type = cfg["planning_type"]

    def plan_next_view(self):
        view_list = np.empty((self.num_candidates, 2))

        if self.planning_type == "local":
            for i in range(self.num_candidates):
                view_list[i] = random_view(
                    self.current_pose[:3, 3],
                    self.radius,
                    self.phi_min,
                    min_view_change=0.2,
                    max_view_change=self.view_change,
                )
        elif self.planning_type == "global":
            for i in range(self.num_candidates):
                view_list[i] = uniform_sampling(self.radius, self.phi_min)

        nbv_index = np.random.choice(len(view_list))
        nbv = view_list[nbv_index]
        return nbv
