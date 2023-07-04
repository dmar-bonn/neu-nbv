from planner.planner import Planner
from planner.utils import view_to_pose_batch, random_view, uniform_sampling
import numpy as np
from scipy.spatial import distance


class MaxDistancePlanner(Planner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_candidates = cfg["num_candidates"]
        self.view_change = cfg["view_change"]
        self.planning_type = cfg["planning_type"]

    def get_camera_view_direction(self, poses):
        view_direction = poses[..., :3, 0]
        view_direction = view_direction / np.linalg.norm(view_direction)
        return view_direction

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

        pose_list = view_to_pose_batch(view_list, self.radius)
        new_view_list = self.get_camera_view_direction(pose_list)

        reference_pose_list = self.trajectory[: self.step]
        reference_view_list = self.get_camera_view_direction(reference_pose_list)

        dist_list = []
        for view in new_view_list:
            dist = 0
            count = 0
            for ref_view in reference_view_list:
                # print(view, ref_view)
                cos_dist = distance.cosine(view, ref_view)
                if cos_dist < 0.6:
                    dist += cos_dist
                    count += 1
            # print(dist)
            dist_list.append(dist / count)

        nbv = view_list[np.argmax(dist_list)]
        return nbv
