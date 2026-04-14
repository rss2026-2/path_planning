import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid
import numpy as np
import csv
import time
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
import cv2

sample_sizes = [5000]

class TrajectoryAnalyzer(Node):
    def __init__(self):
        super().__init__('trajectory_analyzer')
        self.goal_time = None
        self.dist_map = None  # will be populated from /map
        self.resolution = None
        self.origin_x = None
        self.origin_y = None
        self.map_yaw = None

        self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, 1)

        volatile_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.RELIABLE
        )
        for n in sample_sizes:
            self.create_subscription(
                PoseArray,
                f'/trajectory/n{n}',
                lambda msg, n=n: self.analyze(msg, n),
                volatile_qos
            )

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

    def goal_cb(self, msg):
        self.goal_time = time.perf_counter()

    def analyze(self, msg, n):
        elapsed = (time.perf_counter() - self.goal_time) * 1000 if self.goal_time else None

        pts = [(p.position.x, p.position.y) for p in msg.poses]
        dist = sum(
            np.linalg.norm(np.array(pts[i]) - np.array(pts[i+1]))
            for i in range(len(pts) - 1)
        )

        # Safety metrics
        if self.dist_map is not None:
            clearances = [
                self.dist_map[self.world_to_grid(x, y)[1],
                              self.world_to_grid(x, y)[0]]
                for x, y in pts
            ]
            min_clearance = min(clearances)
            avg_clearance = sum(clearances) / len(clearances)
        else:
            min_clearance = avg_clearance = None
            self.get_logger().warn('Distance map not yet received, skipping safety metrics.')

        with open('prm_results.csv', 'a', newline='') as f:
            csv.writer(f).writerow([n, dist, len(pts), elapsed, min_clearance, avg_clearance])

        self.get_logger().info(
            f'n={n}: {len(pts)} waypoints, dist={dist:.2f}m, '
            f'time={elapsed:.1f}ms, '
            f'min_clearance={min_clearance:.2f}m, avg_clearance={avg_clearance:.2f}m'
        )

def main():
    rclpy.init()
    rclpy.spin(TrajectoryAnalyzer())
    rclpy.shutdown()
