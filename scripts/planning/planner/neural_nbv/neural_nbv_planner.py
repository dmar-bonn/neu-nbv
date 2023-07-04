import numpy as np
from scipy.spatial.transform import Rotation as R
from planner.planner import Planner
from planner.utils import view_to_pose_batch, random_view, uniform_sampling
from neural_rendering.evaluation.pretrained_model import PretrainedModel
import torch
from dotmap import DotMap
from neural_rendering.utils import util
import torch.nn.functional as F
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import time
import yaml
import os


class NeuralNBVPlanner(Planner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = cfg["device"]
        self.gpu_id = list(map(int, cfg["gpu_id"].split()))
        self.init_sensor_model(cfg)

        self.image_to_tensor = util.get_image_to_tensor_balanced()
        self.num_candidates = cfg["num_candidates"]
        self.sample_type = cfg["sample_type"]
        self.view_change = cfg["view_change"]
        self.local_view_change = cfg["local_view_change"]
        self.selection_range = cfg["selection_range"]
        self.hierachical_sampling = cfg["use_hierachical_sampling"]
        self.sample_ratio = cfg["sample_ratio"]
        self.K = cfg["top_k"]

        self.max_ref_num = cfg["maximal_ref"]
        self.reward_type = cfg["reward_type"]
        self.render_batch_size = cfg["render_batch_size"]
        self.uncertainty_th = cfg["uncertainty_threshold"]

        self.candidate_views = None
        self.candidate_poses = None
        self.render_pairs = None
        self.trajectory_kdtree = None

        # self.depth_for_renderer = torch.empty(
        #     (self.planning_budget, self.H, self.W)
        # ).to(self.device)

    def init_sensor_model(self, cfg):
        assert os.path.exists(cfg["config_path"])
        assert os.path.exists(cfg["checkpoint_path"])

        with open(cfg["config_path"], "r") as config_file:
            model_cfg = yaml.safe_load(config_file)["model"]

        ckpt_file = torch.load(cfg["checkpoint_path"])
        self.model = PretrainedModel(model_cfg, ckpt_file, self.device, self.gpu_id)

        # original image format
        H, W = self.camera_info["image_resolution"]  # (H, W)
        focal = self.camera_info["focal"]  # (f_x, f_y)
        c = self.camera_info["c"]  # (c_x, c_y)

        # desired image format for redendering input
        render_info = cfg["render_info"]
        H_ref, W_ref = render_info["ref_image_resolution"]
        ref_focal = [0, 0]
        ref_c = [0, 0]

        if np.any([H, W] != [H_ref, W_ref]):
            scale_h = H_ref / H
            scale_w = W_ref / W
            ref_focal[0] = scale_w * focal[0]
            ref_focal[1] = scale_h * focal[1]
            ref_c[0] = scale_w * c[0]
            ref_c[1] = scale_h * c[1]

        self.ref_focal = torch.tensor(ref_focal, dtype=torch.float32).to(self.device)
        self.ref_c = torch.tensor(ref_c, dtype=torch.float32).to(self.device)
        self.ref_image_resolution = (H_ref, W_ref)

        self.trajectory_for_renderer = torch.empty((self.planning_budget, 4, 4)).to(
            self.device
        )
        self.rgb_for_renderer = torch.empty((self.planning_budget, 3, H_ref, W_ref)).to(
            self.device
        )

        # desired image format for redendering output
        render_scale = render_info["render_scale"]
        self.H_render = int(render_scale * H_ref)
        self.W_render = int(render_scale * W_ref)
        render_scale = torch.tensor(
            [
                self.W_render / W_ref,
                self.H_render / H_ref,
            ]
        ).to(self.device)
        self.render_focal = render_scale * self.ref_focal
        self.render_c = render_scale * self.ref_c
        self.z_near, self.z_far = render_info["scene_range"]

    def render_novel_views(self, candidate_poses):
        candidate_num = len(candidate_poses)
        reward_list = np.zeros(candidate_num)

        distance_all, ref_index_all = self.trajectory_kdtree.query(
            candidate_poses[:, :3, 3], np.minimum(self.max_ref_num, self.step)
        )
        # distance_all = torch.tensor(distance_all)
        # ref_index_all = torch.tensor(ref_index_all)
        bool_mask = ~np.isinf(distance_all)

        novel_poses = util.coordinate_transformation(
            candidate_poses, format="normal"
        ).to(self.device)

        # render novel view in batch
        split_novel_view = torch.split(
            torch.arange(candidate_num), self.render_batch_size, dim=0
        )

        for i in split_novel_view:
            ref_index = torch.tensor(ref_index_all[i] * bool_mask[i])
            ref_images = self.rgb_for_renderer[ref_index]

            ref_poses = self.trajectory_for_renderer[ref_index]

            render_results = self.rendering(ref_images, ref_poses, novel_poses[i])
            reward_list[i] = self.cal_reward(render_results)

        return reward_list

    def rendering(self, ref_images, ref_poses, novel_poses):
        NP = len(novel_poses)

        with torch.no_grad():
            self.model.network.encode(
                ref_images,
                ref_poses,
                self.ref_focal.unsqueeze(0),
                self.ref_c.unsqueeze(0),
            )
            target_rays = util.gen_rays(
                novel_poses,
                self.W_render,
                self.H_render,
                self.render_focal,
                self.z_near,
                self.z_far,
                self.render_c,
            )  # (IN, H, W, 8)

            target_rays = target_rays.reshape(NP, self.H_render * self.W_render, -1)

            predict = DotMap(self.model.renderer_par(target_rays))
        return predict

    def cal_reward(self, render_results):
        uncertainty = render_results["uncertainty"]

        reward = torch.mean(uncertainty**2, dim=-1).cpu().numpy()
        reward = np.log10(reward)
        return reward

    # one stage planning
    def start_planning(self):
        candidate_views, candidate_poses = self.local_sampling(
            self.num_candidates, self.current_pose[:3, 3], view_change=self.view_change
        )
        reward_list = self.render_novel_views(candidate_poses)
        nbv_index = np.argmax(reward_list)
        return candidate_views[nbv_index]

    def global_sampling(self, num):
        view_list = np.empty((num, 2))

        for i in range(num):
            view_list[i] = uniform_sampling(self.radius, self.phi_min)

        pose_list = view_to_pose_batch(view_list, self.radius)
        return view_list, pose_list

    def local_sampling(self, num, xyz, view_change, min_view_change=0.2):
        view_list = np.empty((num, 2))

        for i in range(num):
            view_list[i] = random_view(
                xyz, self.radius, self.phi_min, min_view_change, view_change
            )

        pose_list = view_to_pose_batch(view_list, self.radius)
        return view_list, pose_list

    def plan_next_view(self):
        import time

        if self.step > 1:
            t1 = time.time()
            nbv = self.start_planning()
            t2 = time.time()
            print((t2 - t1))
            return nbv

        # need at least two views to start the planning
        else:
            random_next_view = random_view(
                self.current_pose[:3, 3],
                self.radius,
                self.phi_min,
                self.view_change - 0.1,
                self.view_change,
            )
            return random_next_view

    def record_trajectory(self, view, pose):
        self.view_trajectory[self.step] = view
        self.trajectory[self.step] = pose

        # maintain current measurment positions in kd tree
        self.trajectory_kdtree = spatial.KDTree(self.trajectory[: self.step + 1, :3, 3])

        self.trajectory_for_renderer[self.step] = util.coordinate_transformation(
            pose, format="normal"
        ).to(self.device)

    def record_rgb_measurement(self, rgb):
        rgb = np.clip(rgb, a_min=0, a_max=255)
        rgb = rgb / 255
        self.rgb_measurements[self.step] = rgb

        ref_image = self.image_to_tensor(rgb).to(self.device)
        ref_image = F.interpolate(
            ref_image.unsqueeze(0), size=self.ref_image_resolution, mode="area"
        ).squeeze(0)
        self.rgb_for_renderer[self.step] = ref_image

    def test_visualize(self, ref_images, results_dict):
        import matplotlib.pyplot as plt

        H = 60
        W = 60

        for i in range(self.render_batch_size):
            rgb = results_dict.rgb[i].cpu().numpy().reshape(H, W, 3)
            depth = results_dict.depth[i].cpu().numpy().reshape(H, W)
            uncertainty = results_dict.uncertainty[i].cpu().numpy().reshape(H, W)

            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(rgb)
            axs[1].imshow(uncertainty)
            axs[2].imshow(depth)

            plt.show()
