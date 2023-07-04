import rospy
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from . import utils
import numpy as np


class SimulatorBridge:
    def __init__(self, cfg):
        self.cv_bridge = CvBridge()
        self.current_rgb = None
        self.current_depth = None
        self.camera_type = cfg["camera_type"]
        self.sensor_noise = cfg["sensor_noise"]

        self.get_simulator_camera_info()

        self.pose_pub = rospy.Publisher(
            "/gazebo/set_model_state", ModelState, queue_size=1, latch=True
        )

        if self.camera_type == "rgb_camera":
            self.rgb_sub = rospy.Subscriber(
                "/rgb_camera/rgb_image_raw", Image, self.update_rgb
            )
        elif self.camera_type == "rgbd_camera":
            self.rgb_sub = rospy.Subscriber(
                "/rgbd_camera/rgb_image_raw", Image, self.update_rgb
            )
            self.depth_sub = rospy.Subscriber(
                "/rgbd_camera/depth_image_raw", Image, self.update_depth
            )

    def get_simulator_camera_info(self):
        camera_info_raw = rospy.wait_for_message(
            f"/{self.camera_type}/camera_info", CameraInfo
        )
        K = camera_info_raw.K  # intrinsic matrix
        H = int(camera_info_raw.height)  # image height
        W = int(camera_info_raw.width)  # image width
        self.camera_info = {
            "image_resolution": [H, W],
            "c": [K[2], K[5]],
            "focal": [K[0], K[4]],
        }

    def move_camera(self, pose):
        quaternion = utils.rotation_2_quaternion(pose[:3, :3])
        translation = pose[:3, -1]

        camera_pose_msg = ModelState()
        camera_pose_msg.model_name = self.camera_type
        camera_pose_msg.pose.position.x = translation[0]
        camera_pose_msg.pose.position.y = translation[1]
        camera_pose_msg.pose.position.z = translation[2]
        camera_pose_msg.pose.orientation.x = quaternion[0]
        camera_pose_msg.pose.orientation.y = quaternion[1]
        camera_pose_msg.pose.orientation.z = quaternion[2]
        camera_pose_msg.pose.orientation.w = quaternion[3]

        self.pose_pub.publish(camera_pose_msg)

    def update_rgb(self, data):
        self.current_rgb = data

    def update_depth(self, data):
        self.current_depth = data

    def get_image(self):
        rgb = self.cv_bridge.imgmsg_to_cv2(self.current_rgb, "rgb8")
        rgb = np.array(rgb, dtype=float)

        if self.sensor_noise != 0:
            noise = np.random.normal(0.0, self.sensor_noise, rgb.shape)
            rgb += noise

        if self.camera_type == "rgb_camera":
            depth = None
        elif self.camera_type == "rgbd_camera":
            depth = self.cv_bridge.imgmsg_to_cv2(self.current_depth, "32FC1")

        return np.asarray(rgb), np.asarray(depth)
