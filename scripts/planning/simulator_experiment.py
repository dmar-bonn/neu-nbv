import rospy
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

import yaml
import argparse
from planner import get_planner
from planner.utils import uniform_sampling
import numpy as np
import scipy.spatial as spatial
from datetime import datetime
import imageio
import glob
from dotmap import DotMap
import torch
from neural_rendering.utils import util
from neural_rendering.evaluation.pretrained_model import PretrainedModel
import pandas
import torch.nn.functional as F

planner_title = {
    "max_distance": "Max. View Distance",
    "random": "Random",
    "neural_nbv": "Ours",
}


def setup_random_seed(seed):
    np.random.seed(seed)


def main():
    # planning experiment in simulator using baseline planners and our planner

    setup_random_seed(10)

    rospy.init_node("simulator_experiment")
    args = parse_args()
    planner_type_list = ["max_distance", "random", "neural_nbv"]

    repeat = args.repeat
    experiment_path = args.experiment_path

    if not args.evaluation_only:
        experiment_path = os.path.join(
            root_dir,
            "experiments",
            "simulator",
            datetime.now().strftime("%d-%m-%Y-%H-%M"),
        )
        os.makedirs(experiment_path, exist_ok=True)

        print("---------- planning ----------")
        for i in range(repeat):
            # initialize planning with 2 same views
            random_initial_view = []
            for _ in range(2):
                random_initial_view.append(
                    uniform_sampling(radius=2, phi_min=0.15)
                )  # hard-coded, should be the same for config file

            for planner_type in planner_type_list:
                # find planner configuration file
                print(
                    f"---------- {planner_type} planner, experiment ID {i} ----------\n"
                )
                planner_cfg_path = os.path.join(
                    "planning/config", f"{planner_type}_planner.yaml"
                )
                assert os.path.exists(planner_cfg_path)
                with open(planner_cfg_path, "r") as config_file:
                    planner_cfg = yaml.safe_load(config_file)

                planner_cfg.update(args.__dict__)
                planner_cfg["planner_type"] = planner_type
                planner_cfg["experiment_path"] = experiment_path
                planner_cfg["experiment_id"] = i

                nbv_planner = get_planner(planner_cfg)
                nbv_planner.start(initial_view=random_initial_view)

    print("---------- evaluation ----------")
    gpu_id = list(map(int, args.gpu_id.split()))
    device = util.get_cuda(gpu_id[0])

    log_path = os.path.join(root_dir, "neural_rendering", "logs", args.model_name)
    assert os.path.exists(log_path), "experiment does not exist"
    with open(f"{log_path}/training_setup.yaml", "r") as config_file:
        cfg = yaml.safe_load(config_file)

    checkpoint_path = os.path.join(log_path, "checkpoints", "best.ckpt")
    assert os.path.exists(checkpoint_path), "checkpoint does not exist"
    ckpt_file = torch.load(checkpoint_path)

    model = PretrainedModel(cfg["model"], ckpt_file, device, gpu_id)

    # load test view data as ground truth
    test_rgbs, test_poses, focal, c = get_image_data(
        args.test_data_path, "normal", device
    )

    # configure rendering information
    nview = int(args.nviews)
    _, _, H, W = test_rgbs.shape

    z_near = cfg["data"]["dataset"]["z_near"]
    z_far = cfg["data"]["dataset"]["z_far"]

    step_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    total_df = pandas.DataFrame(
        {
            "Planning Type": [],
            "Reference Image Num.": [],
            "PSNR": [],
            "SSIM": [],
        }
    )
    for r in range(repeat):
        for planner_type in planner_type_list:
            ref_data_path = os.path.join(experiment_path, planner_type, str(r))
            ref_rgbs, ref_poses, _, _ = get_image_data(ref_data_path, "normal", device)

            for step in step_list:
                print(
                    f"---------- planner:{planner_type}, repeat {r}, step {step} ----------\n"
                )
                ref_kd_tree = spatial.KDTree(ref_poses[:step, :3, 3].cpu().numpy())
                psnr_list = []
                ssim_list = []
                with torch.no_grad():
                    for i, rgb in enumerate(test_rgbs):
                        pose = test_poses[i]
                        gt = rgb * 0.5 + 0.5
                        gt = gt.permute(1, 2, 0).cpu().numpy()

                        _, ref_index = ref_kd_tree.query(
                            pose[:3, 3].cpu().numpy(), np.minimum(nview, step)
                        )

                        model.network.encode(
                            ref_rgbs[ref_index].unsqueeze(0),
                            ref_poses[ref_index].unsqueeze(0),
                            focal.unsqueeze(0),
                            c.unsqueeze(0),
                        )

                        target_rays = util.gen_rays(
                            pose.unsqueeze(0), W, H, focal, z_near, z_far, c
                        )
                        target_rays = target_rays.reshape(1, H * W, -1)
                        predict = DotMap(model.renderer_par(target_rays))
                        metrics_dict = util.calc_metrics(predict, torch.tensor(gt))
                        psnr_list.append(metrics_dict["psnr"])
                        ssim_list.append(metrics_dict["ssim"])

                    psnr_mean = np.mean(psnr_list)
                    ssim_mean = np.mean(ssim_list)
                    print("psnr:", psnr_mean, "ssim:", ssim_mean)
                    dataframe = pandas.DataFrame(
                        {
                            "Planning Type": planner_title[planner_type],
                            "Reference Image Num.": step,
                            "PSNR": psnr_mean,
                            "SSIM": ssim_mean,
                        },
                        index=[r],
                    )

                    total_df = total_df.append(dataframe)

    total_df.to_csv(f"{experiment_path}/dataframe.csv")


image_to_tensor = util.get_image_to_tensor_balanced()


def get_image_data(data_path, coordinate_format, device, rescale=0.5):
    assert os.path.exists(data_path)
    rgb_paths = [
        x
        for x in glob.glob(f"{data_path}/images/*")
        if (x.endswith(".jpg") or x.endswith(".png"))
    ]
    rgb_paths = sorted(rgb_paths)

    images = []
    poses = []

    for image_path in rgb_paths:
        image = imageio.imread(image_path)[..., :3]

        image = image_to_tensor(image)

        images.append(image)

    pose_list = np.load(f"{data_path}/trajectory.npy")
    for pose in pose_list:
        pose = util.coordinate_transformation(pose, format=coordinate_format)
        poses.append(pose)

    with open(f"{data_path}/camera_info.yaml") as file:
        intrinsic = yaml.safe_load(file)

    images = torch.stack(images).to(device)
    poses = torch.stack(poses).to(device)

    if rescale != 1:
        _, _, H, W = images.shape
        H = int(rescale * H)
        W = int(rescale * W)
        images = F.interpolate(images, size=[W, H], mode="area")

    focal = rescale * torch.tensor(intrinsic["focal"], dtype=torch.float32).to(device)
    c = rescale * torch.tensor(intrinsic["c"], dtype=torch.float32).to(device)

    assert len(images) == len(poses)

    return images, poses, focal, c


def test_visualize(results_dict):
    import matplotlib.pyplot as plt

    H = 400
    W = 400

    rgb = results_dict.rgb[0].cpu().numpy().reshape(H, W, 3)
    depth = results_dict.depth[0].cpu().numpy().reshape(H, W)
    uncertainty = results_dict.uncertainty[0].cpu().numpy().reshape(H, W)
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(rgb)
    axs[1].imshow(uncertainty)
    axs[2].imshow(depth)

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        "-M",
        type=str,
        required=True,
        help="model name of pretrained model",
    )

    parser.add_argument(
        "--test_data_path",
        "-TD",
        type=str,
        required=True,
        help="data path",
    )

    # mandatory arguments
    parser.add_argument(
        "--repeat",
        "-rp",
        type=int,
        default=10,
        help="repeat experiment",
    )
    # arguments with default values
    parser.add_argument(
        "--nviews", "-nv", type=int, default=5, help="number of reference views"
    )
    parser.add_argument(
        "--planning_budget",
        "-BG",
        type=int,
        default=20,
        help="maximal measurments for the mission",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="config file path",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="gpu to use, space delimited",
    )

    parser.add_argument(
        "--evaluation_only", action="store_true", help="evaluation mode"
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        default="not defined",
        help="must be defined in evaluation mode",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
