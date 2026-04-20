import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import csv
import time
import math
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
import cv2
from path_planning.utils import LineTrajectory


class PathAnalyzer(Node):
    """
    Class for comparing two trajectories published by two nodes
    """
    def __init__(self):
        super().__init__('path_analyzer')
        self.declare_parameter("traj_topics", 
                               "['/trajectory/grid_search_trajectory','/trajectory/sampling_trajectory']")

        self.declare_parameter("traj_names", "['Grid Search','Sampling']")
        
        self.traj_topics = self.get_parameter("traj_topics").get_parameter_value().string_array_value
        self.traj_names = self.get_parameter("traj_names").get_parameter_value().string_array_value

        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 10)
        self.pose_sub = self.create_subscription(Odometry, self.odom_topic, self.pose_cb, 10)

        for traj_topic, traj_name in zip(self.traj_topics, self.traj_names):
            self.create_subscription(
                PoseArray, 
                traj_topic,
                lambda traj_msg, name = traj_name: self.traj_cb(traj_msg, name),
                10
            )

        self.start_time = None
        self.pose = None

        self.data = {"goal_distances": []}
        for traj_name in self.traj_names:
            self.data[traj_name] = {
                "computation_times": [],
                "path_distances": []
            }
        self.get_logger().info("Ready to start analysis.")
    
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
        dist_to_goal = int(math.dist(start_pt, end_pt))
        self.data["goal_distances"].append(dist_to_goal)
        self.get_logger().info(f"New Goal Received! Minimum Distance = {dist_to_goal}\nWaiting for Trajectories...")
    
    def traj_cb(self, traj_msg, traj_name):
        computation_time = int((time.perf_counter_ns() - self.start_time) / 1e6)
        new_traj = LineTrajectory(self)
        new_traj.fromPoseArray(traj_msg)
        path_dist = new_traj.distance_to_end(0)

        self.data[traj_name]["computation_times"].append(computation_time)
        self.data[traj_name]["path_distances"].append(path_dist)

        self.get_logger().info(f"{traj_name}:\n\tComputation Time: {computation_time} ms.\n\tPath Distance: {path_dist}")
