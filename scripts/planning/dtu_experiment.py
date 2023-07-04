import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from neural_rendering.evaluation.pretrained_model import PretrainedModel
from neural_rendering.data import get_data
from neural_rendering.utils import parser, util
import yaml
from dotmap import DotMap
import torch
import warnings
import numpy as np
import pandas
import seaborn as sb
import copy
from scipy.spatial import distance
from datetime import datetime
import random
import pickle
from dotmap import DotMap


warnings.filterwarnings("ignore")

# follow pixelnerf setup
candidate_index_list = [
    6,
    7,
    8,
    9,
    10,
    13,
    14,
    15,
    16,
    17,
    21,
    22,
    23,
    24,
    25,
    31,
    32,
    33,
    34,
    35,
    41,
    42,
    43,
    44,
    45,
]


def setup_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_nbv_ref_index(
    model, images, poses, focal, c, z_near, z_far, candidate_list, budget, ref_index
):
    _, _, H, W = images.shape

    for i in range(budget):
        remain_candidate_list = list(set(candidate_list) - set(ref_index))
        reward_list = []

        model.network.encode(
            images[ref_index].unsqueeze(0),
            poses[ref_index].unsqueeze(0),
            focal.unsqueeze(0),
            c.unsqueeze(0),
        )

        for target_view in remain_candidate_list:
            novel_pose = poses[target_view]
            target_rays = util.gen_rays(
                novel_pose.unsqueeze(0), W, H, focal, z_near, z_far, c
            )
            target_rays = target_rays.reshape(1, H * W, -1)
            predict = DotMap(model.renderer_par(target_rays))
            uncertainty = predict["uncertainty"][0]
            reward = torch.sum(uncertainty**2).cpu().numpy()
            reward_list.append(reward)

        nbv_index = np.argmax(reward_list)
        new_ref_index = remain_candidate_list[nbv_index]
        ref_index.append(new_ref_index)

    return ref_index


def get_camera_view_direction(poses):
    poses = poses.cpu().numpy()
    view_direction = -poses[..., :3, 2]
    view_direction = view_direction / np.linalg.norm(view_direction)
    return view_direction


def get_max_dist_ref_index(poses, ref_index, candidate_list, budget):
    view_direction = get_camera_view_direction(poses)

    for i in range(budget):
        remain_candidate_list = list(set(candidate_list) - set(ref_index))
        cos_distance_list = []

        for idx in remain_candidate_list:
            cos_dist = 0.0
            for image_idx in ref_index:
                cos_dist += distance.cosine(
                    view_direction[idx], view_direction[image_idx]
                )
            cos_distance_list.append(cos_dist)
        new_ref_index = remain_candidate_list[np.argmax(cos_distance_list)]
        ref_index.append(new_ref_index)
    return ref_index


def main():
    # planning experiment on DTU using baseline planners and our planner
    setup_random_seed(10)
    args = parser.parse_args(planning_args)
    dtu_nbv_planner = DTUNBVPlanning(args)

    experiment_path = args.experiment_path

    if args.evaluation_only:
        with open(f"{experiment_path}/saved_index_dict.pkl", "rb") as f:
            index_record = pickle.load(f)
    else:
        experiment_path = os.path.join(
            root_dir,
            "experiments",
            "dtu",
            datetime.now().strftime("%d-%m-%Y-%H-%M"),
        )
        os.makedirs(experiment_path)
        index_record = dtu_nbv_planner.planning()
        with open(f"{experiment_path}/saved_index_dict.pkl", "wb") as f:
            pickle.dump(index_record, f)

    total_df = dtu_nbv_planner.evaluation(index_record)
    total_df.to_csv(f"{experiment_path}/dataframe.csv")


class DTUNBVPlanning:
    """
    planning on DTU using different view selection methods: max_view_distance, random, and our uncertainty guided
    """

    def __init__(self, args):
        log_path = os.path.join(root_dir, "neural_rendering", "logs", args.model_name)
        assert os.path.exists(log_path), "experiment does not exist"
        with open(f"{log_path}/training_setup.yaml", "r") as config_file:
            cfg = yaml.safe_load(config_file)

        checkpoint_path = os.path.join(log_path, "checkpoints", "best.ckpt")
        assert os.path.exists(checkpoint_path), "checkpoint does not exist"
        ckpt_file = torch.load(checkpoint_path)

        gpu_id = list(map(int, args.gpu_id.split()))
        self.device = util.get_cuda(gpu_id[0])

        self.repeat = args.repeat
        self.model = PretrainedModel(cfg["model"], ckpt_file, self.device, gpu_id)

        cfg["data"]["dataset"]["data_rootdir"] = os.path.join(
            root_dir, "neural_rendering/data/dataset/dtu_dataset/rs_dtu_4/DTU"
        )
        datamodule = get_data(cfg["data"])
        self.dataset = datamodule.load_dataset("val")
        self.z_near = self.dataset.z_near
        self.z_far = self.dataset.z_far

    def planning(self):
        print(f"---------- planning ---------- \n")
        ON = len(self.dataset)

        selection_type = ["Max. View Distance", "Random", "Ours"]
        nview_list = [2, 3, 4, 5, 6, 7, 8, 9]  # maximal budget = 9
        scene_index = range(ON)

        ref_index_record = {}

        with torch.no_grad():
            for nviews in nview_list:
                ref_index_record[nviews] = {}
                print(f"---------- {nviews} views experiment---------- \n")
                for i in scene_index:
                    data_instance = self.dataset.__getitem__(i)
                    scene_title = data_instance["scan_name"]
                    ref_index_record[nviews][i] = {}

                    print(f"test on {scene_title}")
                    images = data_instance["images"].to(self.device)
                    focal = data_instance["focal"].to(self.device)
                    c = data_instance["c"].to(self.device)
                    poses = data_instance["poses"].to(self.device)

                    # random initialize first 2 ref images for all methods
                    for r in range(self.repeat):
                        ref_index_record[nviews][i][r] = {}
                        initial_ref_index = list(
                            np.random.choice(candidate_index_list, 2, replace=False)
                        )

                        candidate_list = list(
                            set(candidate_index_list) - set(initial_ref_index)
                        )
                        budget = nviews - 2

                        for stype in selection_type:
                            print(f"---------- repeat: {r}, {stype} ---------- \n")

                            if stype == "Max. View Distance":
                                ref_index = get_max_dist_ref_index(
                                    poses,
                                    copy.deepcopy(initial_ref_index),
                                    candidate_list,
                                    budget,
                                )

                                print(ref_index)
                            elif stype == "Random":
                                random_ref_index = list(
                                    np.random.choice(
                                        candidate_index_list, budget, replace=True
                                    )
                                )
                                ref_index = initial_ref_index + random_ref_index
                                print(ref_index)
                                ref_index = np.unique(ref_index)

                            elif stype == "Ours":
                                ref_index = get_nbv_ref_index(
                                    self.model,
                                    images,
                                    poses,
                                    focal,
                                    c,
                                    self.z_near,
                                    self.z_far,
                                    candidate_list,
                                    budget,
                                    copy.deepcopy(initial_ref_index),
                                )
                                print(ref_index)

                            ref_index_record[nviews][i][r][stype] = ref_index

        return ref_index_record

    def evaluation(self, index_record):
        print(f"---------- evaluation ---------- \n")
        total_df = pandas.DataFrame(
            {
                "Planning Type": [],
                "Reference Image Number": [],
                "PSNR": [],
                "SSIM": [],
                "Scene": [],
            }
        )
        with torch.no_grad():
            for nviews, nviews_dict in index_record.items():
                print(f"---------- {nviews} views experiment---------- \n")
                for scene_id, scene_dict in nviews_dict.items():
                    data_instance = self.dataset.__getitem__(scene_id)
                    scene_title = data_instance["scan_name"]

                    print(f"test on {scene_title}")
                    images = data_instance["images"].to(self.device)
                    images_0to1 = images * 0.5 + 0.5
                    _, _, H, W = images.shape
                    focal = data_instance["focal"].to(self.device)
                    c = data_instance["c"].to(self.device)
                    poses = data_instance["poses"].to(self.device)
                    psnr_per_scene = []
                    ssim_per_scene = []

                    # random initialize first 2 ref images for all methods
                    for repeat, repeat_dict in scene_dict.items():
                        for stype, ref_index in repeat_dict.items():
                            print(f"---------- repeat: {repeat}, {stype} ---------- \n")
                            print(ref_index)
                            self.model.network.encode(
                                images[ref_index].unsqueeze(0),
                                poses[ref_index].unsqueeze(0),
                                focal.unsqueeze(0),
                                c.unsqueeze(0),
                            )
                            test_index = list(
                                set(candidate_index_list) - set(ref_index)
                            )
                            psnr_per_test = []
                            ssim_per_test = []

                            for target_view in test_index:
                                gt = (
                                    images_0to1[target_view]
                                    .permute(1, 2, 0)
                                    .cpu()
                                    .numpy()
                                )
                                novel_pose = poses[target_view]
                                target_rays = util.gen_rays(
                                    novel_pose.unsqueeze(0),
                                    W,
                                    H,
                                    focal,
                                    self.z_near,
                                    self.z_far,
                                    c,
                                )
                                target_rays = target_rays.reshape(1, H * W, -1)

                                predict = DotMap(self.model.renderer_par(target_rays))

                                metrics_dict = util.calc_metrics(
                                    predict, torch.tensor(gt)
                                )
                                psnr_per_test.append(metrics_dict["psnr"])
                                ssim_per_test.append(metrics_dict["ssim"])

                            psnr_per_scene = np.mean(psnr_per_test)
                            ssim_per_scene = np.mean(ssim_per_test)
                            print(psnr_per_scene, ssim_per_scene)

                            dataframe = pandas.DataFrame(
                                {
                                    "Planning Type": stype,
                                    "Reference Image Number": nviews,
                                    "PSNR": psnr_per_scene,
                                    "SSIM": ssim_per_scene,
                                    "Scene": scene_id,
                                },
                                index=[repeat],
                            )

                            total_df = total_df.append(dataframe)
        return total_df


def planning_args(parser):
    """
    Parse arguments for evaluation setup.
    """

    parser.add_argument(
        "--model_name",
        "-M",
        type=str,
        required=True,
        help="model name of pretrained model",
    )

    parser.add_argument(
        "--repeat",
        "-R",
        type=int,
        default=5,
        help="repeat times for planning experiment",
    )

    # arguments with default values
    parser.add_argument(
        "--evaluation_only", action="store_true", help="evaluation mode"
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        default="not defined",
        help="must be defined in evaluation mode",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="GPU(s) to use, space delimited"
    )
    return parser


if __name__ == "__main__":
    main()
