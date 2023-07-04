from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
import json

# from rembg import remove
import os
import imageio


def get_roi_mask(rgb):
    """binary mask for ROIs using color thresholding"""
    hsv = cv2.cvtColor(np.array(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([20, 255, 255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([160, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask0 + mask1
    mask = mask + 1e-5

    return mask


def get_black_mask(rgb):
    """binary mask for ROIs using color thresholding"""
    lower_black = np.array([250, 250, 250])
    upper_black = np.array([255, 255, 255])
    mask = cv2.inRange(rgb, lower_black, upper_black)

    return mask


def visualize_uncertainty(uncertainty):
    variance = np.exp(uncertainty)


def rotation_2_quaternion(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    return r.as_quat()


def xyz_to_view(xyz, radius):
    phi = np.arcsin(xyz[2] / radius)  # phi from 0 to 0.5*pi
    theta = np.arctan2(xyz[1], xyz[0]) % (2 * np.pi)  # theta from 0 to 2*pi

    return [phi, theta]


def view_to_pose(view, radius):
    phi, theta = view

    # phi should be within [min_phi, 0.5*np.pi)
    if phi >= 0.5 * np.pi:
        phi = np.pi - phi

    pose = np.eye(4)
    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.cos(phi)
    z = radius * np.sin(phi)

    translation = np.array([x, y, z])
    rotation = R.from_euler("ZYZ", [theta, -phi, np.pi]).as_matrix()

    pose[:3, -1] = translation
    pose[:3, :3] = rotation
    return pose


def view_to_pose_batch(views, radius):
    num = len(views)
    phi = views[:, 0]
    theta = views[:, 1]

    # phi should be within [min_phi, 0.5*np.pi)
    index = phi >= 0.5 * np.pi
    phi[index] = np.pi - phi[index]

    poses = np.broadcast_to(np.identity(4), (num, 4, 4)).copy()

    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.cos(phi)
    z = radius * np.sin(phi)

    translations = np.stack((x, y, z), axis=-1)

    angles = np.stack((theta, -phi, np.pi * np.ones(num)), axis=-1)
    rotations = R.from_euler("ZYZ", angles).as_matrix()

    poses[:, :3, -1] = translations
    poses[:, :3, :3] = rotations

    return poses


def random_view(current_xyz, radius, phi_min, min_view_change, max_view_change):
    """
    random scatter view direction changes by given current position and view change range.
    """

    u = current_xyz / np.linalg.norm(current_xyz)

    # pick a random vector:
    r = np.random.multivariate_normal(np.zeros_like(u), np.eye(len(u)))

    # form a vector perpendicular to u:
    uperp = r - r.dot(u) * u
    uperp = uperp / np.linalg.norm(uperp)

    # random view angle change in radian
    random_view_change = np.random.uniform(low=min_view_change, high=max_view_change)
    cosine = np.cos(random_view_change)
    w = cosine * u + np.sqrt(1 - cosine**2 + 1e-8) * uperp
    w = radius * w / np.linalg.norm(w)

    view = xyz_to_view(w, radius)

    if view[0] < phi_min:
        view[0] = phi_min

    return view


def uniform_sampling(radius, phi_min):
    """
    uniformly generate unit vector on hemisphere.
    then calculate corresponding view direction targeting coordinate origin.
    """

    xyz = np.array([0.0, 0.0, 0.0])

    # avoid numerical error
    while np.linalg.norm(xyz) < 0.001:
        xyz[0] = np.random.uniform(low=-1.0, high=1.0)
        xyz[1] = np.random.uniform(low=-1.0, high=1.0)
        xyz[2] = np.random.uniform(low=0.0, high=1.0)

    xyz = radius * xyz / np.linalg.norm(xyz)
    view = xyz_to_view(xyz, radius)

    if view[0] < phi_min:
        view[0] = phi_min
    return view


def focal_len_to_fov(focal, resolution):
    """
    calculate FoV based on given focal length adn image resolution

    Args:
        focal: [fx, fy]
        resolution: [W, H]

    Returns:
        FoV: [HFoV, VFoV]

    """
    focal = np.asarray(focal)
    resolution = np.asarray(resolution)

    return 2 * np.arctan(0.5 * resolution / focal)


def mask_out_background(image_path):
    """remove background"""

    rgb = imageio.imread(image_path)
    masked_rgb = remove(rgb)
    # H, W, _ = rgb.shape
    # masked_rgb = np.ones((H, W, 4)) * 255
    # masked_rgb[..., :3] = rgb
    # mask_white = rgb >= np.array([254, 254, 254])
    # mask_white = np.all(mask_white, axis=-1)
    # mask_black = rgb <= np.array([1, 1, 1])
    # mask_black = np.all(mask_black, axis=-1)
    # masked_rgb[mask_white] = [0, 0, 0, 0]
    # masked_rgb[mask_black] = [0, 0, 0, 0]

    return masked_rgb


def record_render_data(path, camera_info, trajectory, use_masked_image=False):
    transformation = np.array(
        [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )  # transform gazebo coordinate to opengl format
    opencv_trajectory = np.empty(trajectory.shape)

    for i, pose in enumerate(trajectory):
        opencv_trajectory[i] = pose @ transformation

    resolution = camera_info["image_resolution"]
    c = camera_info["c"]
    focal = camera_info["focal"]

    fov = focal_len_to_fov(focal, resolution)

    record_dict = {}
    record_dict["camera_angle_x"] = fov[0]
    record_dict["camera_angle_y"] = fov[1]
    record_dict["fl_x"] = focal[0]
    record_dict["fl_y"] = focal[1]
    record_dict["k1"] = 0.000001
    record_dict["k2"] = 0.000001
    record_dict["p1"] = 0.000001
    record_dict["p2"] = 0.000001
    record_dict["cx"] = c[0]
    record_dict["cy"] = c[1]
    record_dict["w"] = resolution[0]
    record_dict["h"] = resolution[1]
    record_dict["frames"] = []
    record_dict["scale"] = 1.0
    record_dict["aabb_scale"] = 2.0

    for i, pose in enumerate(opencv_trajectory):
        image_file = f"images/{i+1:04d}.png"
        image_path = os.path.join(path, image_file)

        if use_masked_image:
            masked_image = mask_out_background(image_path)
            image_file = f"images/masked_{i+1:04d}.png"
            image_path = os.path.join(path, image_file)
            imageio.imwrite(image_path, masked_image)

        data_frame = {
            "file_path": image_file,
            # "sharpness": 30.0,
            "transform_matrix": pose.tolist(),
        }
        record_dict["frames"].append(data_frame)

    with open(f"{path}/transforms.json", "w") as f:
        json.dump(record_dict, f, indent=4)

    # for test only
    # for i, pose in enumerate(opencv_trajectory[50:]):
    #     data_frame = {
    #         "file_path": f"images/{i+51:04d}.jpg",
    #         "sharpness": 30.0,
    #         "transform_matrix": pose.tolist(),
    #     }
    #     record_dict["frames"].append(data_frame)

    # with open(f"{path}/test_transforms.json", "w") as f:
    #     json.dump(record_dict, f, indent=4)


def test():
    view = []
    # for i in range(5):
    #     new_view = uniform_sampling(2, 0.15)
    #     view.append(new_view)
    #     print("view:", new_view)
    current_xyz = [0, 0, 2]
    for i in range(500):
        local = random_view(current_xyz, 2, 0.15, 0.2, 1.05)
        view.append(local)

    xyz_list = view_to_pose_batch(np.array(view), 2)[..., :3, 3]
    print(xyz_list)
    for xyz in xyz_list:
        view = xyz_to_view(xyz, 2)
        print(view)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(xyz_list[..., 0], xyz_list[..., 1], xyz_list[..., 2])
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    ax.set_zlim(0, 2.5)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test()
