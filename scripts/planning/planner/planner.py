from .simulator_bridge import SimulatorBridge
from . import utils
import time
import os
from datetime import datetime
import numpy as np
import imageio.v2 as imageio
import yaml


class Planner:
    def __init__(self, cfg):
        self.simulator_bridge = SimulatorBridge(cfg["simulation_bridge"])
        self.camera_info = self.simulator_bridge.camera_info

        self.record_path = os.path.join(
            cfg["experiment_path"], cfg["planner_type"], str(cfg["experiment_id"])
        )

        self.planning_budget = cfg["planning_budget"]
        self.initial_type = cfg["initial_type"]

        self.H, self.W = self.camera_info[
            "image_resolution"
        ]  # original image resolution from simulator
        self.trajectory = np.empty((self.planning_budget, 4, 4))
        self.view_trajectory = np.empty((self.planning_budget, 2))  # [phi, theta]
        self.rgb_measurements = np.empty((self.planning_budget, self.H, self.W, 3))
        self.depth_measurements = np.empty((self.planning_budget, self.H, self.W))
        self.step = 0

        self.config_actionspace(cfg["action_space"])

    def config_actionspace(self, cfg):
        """set hemisphere actionspace parameters"""

        self.min_height = cfg["min_height"]
        self.radius = cfg["radius"]
        self.phi_min = np.arcsin(self.min_height / self.radius)
        self.phi_max = 0.5 * np.pi
        self.theta_min = 0
        self.theta_max = 2 * np.pi

    def init_camera_pose(self, initial_view):
        print("------ start mission ------ \n")
        print("------ initialize camera pose ------ \n")

        if initial_view is None:
            if self.initial_type == "random":
                initial_view = utils.uniform_sampling(self.radius, self.phi_min)

            elif self.initial_type == "pre_calculated":
                self.get_view_list()
                initial_view = next(self.view_list)

            self.move_sensor(initial_view)

        else:
            for view in initial_view:
                self.move_sensor(view)

    def start(self, initial_view=None):
        self.init_camera_pose(initial_view)

        while self.step < self.planning_budget:
            next_view = self.plan_next_view()
            self.move_sensor(next_view)

        self.record_experiment()
        print("------ complete mission ------\n")
        # rospy.signal_shutdown("shut down ros node")

    def move_sensor(self, view):
        pose = utils.view_to_pose(view, self.radius)
        self.simulator_bridge.move_camera(pose)

        self.current_view = view
        self.current_pose = pose
        print(pose)
        print(
            f"------ reach given pose and take measurement No.{self.step + 1} ------\n"
        )
        time.sleep(1)  # lazy solution to make sure we receive correct images
        rgb, depth = self.simulator_bridge.get_image()
        self.record_step(view, pose, rgb, depth)
        self.step += 1

    def plan_next_view(self):
        raise NotImplementedError("plan_next_view method is not implemented")

    def record_experiment(self):
        print("------ record experiment data ------\n")

        os.makedirs(self.record_path, exist_ok=True)
        images_path = os.path.join(self.record_path, "images")
        os.mkdir(images_path)
        depths_path = os.path.join(self.record_path, "depths")
        os.mkdir(depths_path)

        for i, rgb in enumerate(self.rgb_measurements):
            imageio.imwrite(
                f"{images_path}/{i+1:04d}.png", (rgb * 255).astype(np.uint8)
            )

        if len(self.depth_measurements) > 0:
            for i, depth in enumerate(self.depth_measurements):
                with open(f"{depths_path}/depth_{i+1:04d}.npy", "wb") as f:
                    depth_array = np.array(depth, dtype=np.float32)
                    np.save(f, depth_array)

        with open(f"{self.record_path}/trajectory.npy", "wb") as f:
            np.save(f, self.trajectory)

        with open(f"{self.record_path}/camera_info.yaml", "w") as f:
            yaml.safe_dump(self.camera_info, f)

        # record json data required for instant-ngp training
        utils.record_render_data(self.record_path, self.camera_info, self.trajectory)

    def record_step(self, view, pose, rgb, depth):
        self.record_trajectory(view, pose)
        self.record_rgb_measurement(rgb)
        if depth is not None:
            self.record_depth_measurement(depth)

    def record_rgb_measurement(self, rgb):
        rgb = np.clip(rgb, a_min=0, a_max=255)
        rgb = rgb / 255
        self.rgb_measurements[self.step] = rgb

    def record_depth_measurement(self, depth):
        self.depth_measurements[self.step] = depth

    def record_trajectory(self, view, pose):
        self.view_trajectory[self.step] = view
        self.trajectory[self.step] = pose
