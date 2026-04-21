import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid
import numpy as np

import time
import math


from path_planning.utils import LineTrajectory
from scipy.spatial.transform import Rotation as R
import cv2


class PathAnalyzer(Node):
    """
    Class for comparing two trajectories published by two nodes
    """
    def __init__(self):
        super().__init__('path_analyzer')
        self.declare_parameter("traj_topics", 
                               ['/trajectory/grid_search_trajectory','/trajectory/sampling_trajectory'])
        self.declare_parameter("traj_names", ['Grid Search','Sampling'])
        self.declare_parameter('odom_topic', "/odom")
        
        self.traj_topics = self.get_parameter("traj_topics").get_parameter_value().string_array_value
        self.traj_names = self.get_parameter("traj_names").get_parameter_value().string_array_value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value

        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 10)
        self.pose_sub = self.create_subscription(Odometry, self.odom_topic, self.pose_cb, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, 1)

        for traj_topic, traj_name in zip(self.traj_topics, self.traj_names):
            self.create_subscription(
                PoseArray, 
                traj_topic,
                lambda traj_msg, name = traj_name: self.traj_cb(traj_msg, name),
                10
            )

        self.start_time = None
        self.pose = None

        self.dist_map = None  # will be populated from /map
        self.resolution = None
        self.origin_x = None
        self.origin_y = None
        self.map_yaw = None

        self.data = {"goal_distances": []}
        for traj_name in self.traj_names:
            self.data[traj_name] = {
                "computation_times": [],
                "path_distances": [],
                "min_clearances": [],
                "avg_clearances": []
            }
        self.get_logger().info("Ready to start analysis.")

    def map_cb(self, msg):
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        quat = [msg.info.origin.orientation.x, msg.info.origin.orientation.y,
                msg.info.origin.orientation.z, msg.info.origin.orientation.w]
        self.map_yaw = R.from_quat(quat).as_euler('xyz')[2]

        map_data = np.array(msg.data, np.uint8)
        binary_map = (map_data == 0).astype(np.uint8)
        dist_pixels = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5)
        # Convert from pixels to metres
        self.dist_map = dist_pixels * self.resolution
        self.dist_map = self.dist_map.reshape((msg.info.height, msg.info.width))
        self.get_logger().info('Distance map ready.')
    
    def world_to_grid(self, x, y):
        tx, ty = x - self.origin_x, y - self.origin_y
        cos_q, sin_q = np.cos(-self.map_yaw), np.sin(-self.map_yaw)
        rx = tx * cos_q - ty * sin_q
        ry = tx * sin_q + ty * cos_q
        ix = int(rx / self.resolution)
        iy = int(ry / self.resolution)
        # Clamp to map bounds
        ix = max(0, min(ix, self.dist_map.shape[1] - 1))
        iy = max(0, min(iy, self.dist_map.shape[0] - 1))
        return ix, iy
    
    def find_clearences(self, trajectory):
        if self.dist_map is None:
            self.get_logger().warn('Distance map not yet received, skipping safety metrics.')
            return

        pts = trajectory.points
        clearances = np.array([
            self.dist_map[self.world_to_grid(x, y)[1],
                            self.world_to_grid(x, y)[0]]
            for x, y in pts
        ])
        return (np.min(clearances), np.mean(clearances))
    
    def pose_cb(self, pose_msg):
        self.pose = {
            "position": [
                pose_msg.pose.pose.position.x,
                pose_msg.pose.pose.position.y,
                pose_msg.pose.pose.position.z
            ],
            "orientation": [
                pose_msg.pose.pose.orientation.x,
                pose_msg.pose.pose.orientation.y,
                pose_msg.pose.pose.orientation.z,
                pose_msg.pose.pose.orientation.w,
            ]
        }

    def goal_cb(self, goal_msg):
        self.start_time = time.perf_counter_ns()
        
        start_pt = (self.pose["position"][0], self.pose["position"][1])
        end_pt = (goal_msg.pose.position.x, goal_msg.pose.position.y)
        dist_to_goal = math.dist(start_pt, end_pt)
        self.data["goal_distances"].append(dist_to_goal)
        self.get_logger().info(f"New Goal Received! Minimum Distance = {dist_to_goal}\nWaiting for Trajectories...")
    
    def traj_cb(self, traj_msg, traj_name):
        computation_time = int((time.perf_counter_ns() - self.start_time) / 1e6)
        new_traj = LineTrajectory(self)
        new_traj.fromPoseArray(traj_msg)
        path_dist = new_traj.distance_to_end(0)

        min_clearance, avg_clearance = self.find_clearences(new_traj)

        self.data[traj_name]["computation_times"].append(computation_time)
        self.data[traj_name]["path_distances"].append(path_dist)
        self.data[traj_name]["min_clearances"].append(min_clearance)
        self.data[traj_name]["avg_clearances"].append(avg_clearance)

        self.get_logger().info(f"\n{traj_name}:\n\tComputation Time: {computation_time} ms.\n\t"+
                               f"Path Distance: {path_dist}\n\t"+
                               f"Path Error: {path_dist - self.data['goal_distances'][-1]}\n\t"+
                               f"Min Clearance: {min_clearance}\n\t"+
                               f"Avg Clearance: {avg_clearance}")

def main(args=None):
    rclpy.init(args=args)
    analyzer = PathAnalyzer()
    rclpy.spin(analyzer)
    rclpy.shutdown()