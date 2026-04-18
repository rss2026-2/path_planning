import rclpy

from geometry_msgs.msg import PoseArray, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Odometry
from path_planning.utils import LineTrajectory
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA



import numpy as np
import math
import heapq
from scipy.spatial.transform import Rotation as R
import cv2

import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.search_pub = self.create_publisher(
            MarkerArray,
            "/search_alg",
            10
        )

        self.pose_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")
        self.map = None
        self.pose = None
        
        #reset trials
        self.trials = []
        fig, ax = plt.subplots()
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.trials)))
        for i,trial in enumerate(self.trials):
            ax.plot(trial["x"], trial["y"], marker = "o", color = colors[i], label = trial["label"])

        ax.set_xlabel("Step Size (Cells)")
        ax.set_ylabel("Time (ms)")
        ax.legend(title = "Approx Distance To Goal (Cells)")
        ax.set_title("Relationship between Step Size and Distance from Goal")
        fig.savefig("src/path_planning/path_planning/generated_figs/step_size_trial.png", bbox_inches="tight")
        plt.close(fig)

        self.get_logger().info("Awaiting Map")

    def map_cb(self, map_msg):
        """
        Updates the map used for path planning
        """

        self.get_logger().info("Successfully Received Map Information")

        map_transform = np.eye(4)
        translation = [
            map_msg.info.origin.position.x,
            map_msg.info.origin.position.y,
            map_msg.info.origin.position.z
        ]
        rotation = R.from_quat([
            map_msg.info.origin.orientation.x,
            map_msg.info.origin.orientation.y,
            map_msg.info.origin.orientation.z,
            map_msg.info.origin.orientation.w,
        ])

        map_transform[:3, 3] = translation
        map_transform[:3,:3] = rotation.as_matrix()


        transform_inverse = np.eye(4)
        transform_inverse[:3, :3] = map_transform[:3,:3].T
        transform_inverse[:3, 3] = -map_transform[:3,:3].T @ map_transform[:3, 3]

        occupancy_grid = np.array(map_msg.data).reshape(map_msg.info.height, map_msg.info.width)
        
        safety_cell_radius = 5
        self.map_occupancy_expansion(occupancy_grid, safety_cell_radius)

        self.map = {
            "res": map_msg.info.resolution,
            "transform": map_transform,
            "transform_inv": transform_inverse,
            "array" : occupancy_grid,
            "width" : map_msg.info.width,
            "height": map_msg.info.height
        }

    def map_occupancy_expansion(self, grid, radius, prob_thresh = 0.3):
        # find boundaries of map
        threshold = np.zeros(shape = grid.shape, dtype = np.uint8)
        threshold[grid >= int(prob_thresh * 100)] = 255

        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            for point in contour:
                # point is [[x, y]], so we need to extract the coordinates
                x, y = point[0]
                cv2.circle(threshold, (x, y), radius, color=255, thickness=-1)

        grid[grid == -1] = 1
        grid[threshold > 0] = 1

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
        if self.map is None:
            self.get_logger().info("Map Information Not Received")
            return

        goal_pose = goal_msg.pose
        start_point = (self.pose["position"][0], self.pose["position"][1])
        end_point = (goal_pose.position.x, goal_pose.position.y)

        # self.plan_path(
        #     start_point = start_point,
        #     end_point = end_point
        # )

        # self.get_logger().info("Path Generated!")
        # self.trajectory.publish_viz()

        self.step_size_trial(start_point, end_point)
        pass

    def step_size_trial(self, start_point, end_point):
        new_trial = {
            "label": int(math.dist(start_point, end_point) / self.map["res"]),
            "x" : np.arange(10,21,2),
            "y1": [],
            "y2": []
        }


        for step_size in new_trial["x"]:
            start_time = time.perf_counter_ns()
            self.plan_path(start_point = start_point, end_point = end_point, max_step_size = step_size, visualize = True)
            time_elapsed = int((time.perf_counter_ns() - start_time) / 1e6)
            new_trial["y1"].append(time_elapsed)
            new_trial["y2"].append(self.trajectory.distance_to_end(0))

            self.trajectory.publish_viz()
            time.sleep(2)
        
        self.trials.append(new_trial)

        fig, axs = plt.subplots(1,2, layout='constrained', )
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.trials)))

        
        for i,trial in enumerate(self.trials):
            axs[0].plot(trial["x"], trial["y1"], marker = "o", color = colors[i], label = trial["label"])
            axs[1].plot(trial["x"], trial["y2"], marker = "o", color = colors[i], label = trial["label"])

        axs[0].set_xticks(new_trial["x"])
        axs[0].set_xlabel("Step Size (Cells)")
        axs[0].set_ylabel("Time (ms)")
        axs[0].set_title("Planning Time vs Step Size")

        axs[1].set_xticks(new_trial["x"])
        axs[1].set_xlabel("Step Size (Cells)")
        axs[1].set_ylabel("Distance (meters)")
        axs[1].set_title("Path Distance vs Step Size")

        fig.legend(*axs[0].get_legend_handles_labels(),title = "Approx Distance To Goal (Cells)", loc='upper left', bbox_to_anchor=(1.05, 0.9))
        fig.savefig("src/path_planning/path_planning/generated_figs/step_size_trial.png", bbox_inches="tight")
        plt.close(fig)

        pass


    def plan_path(self, start_point, end_point, max_step_size = 5, visualize = False):
        cells = self.real_to_grid_frame(np.array([start_point, end_point]))
   
        start_cell, end_cell = tuple(cells[0]), tuple(cells[1])
        
        grid_path = self.occupancy_priority_q(start_cell, end_cell, max_step_size, visualize)
        grid_shortened_path = self.shorten_cell_path(grid_path)
        real_path = self.grid_to_real_frame(grid_shortened_path)

        self.trajectory.clear()
        self.trajectory.addPoints(real_path)
    
    def occupancy_priority_q(self, start_cell, end_cell, max_step_size, visualize):
        
        queue = [] # le heap
        seen = set()
        edges = [] # collect all edges for visualization

        start_item = (math.dist(start_cell,end_cell), 0, (start_cell,)) # (total cost, path_cost, path)
        heapq.heappush(queue, start_item)

        map_w, map_h = self.map["width"], self.map["height"]
        def find_valid_neighbors(cell):
            neighbors = []
            curr_u, curr_v = cell

            dist_from_end = math.dist(cell, end_cell)
            step_size = max( min(int(dist_from_end), max_step_size), 1)

            for du,dv in [(1,1), (-1,-1), (1,-1), (-1,1), (-1,0), (1,0), (0,-1), (0,1)]:
                n_cell = (curr_u + (du * step_size), curr_v + (dv*step_size))
                if n_cell not in seen: #if we haven't visited the cell already
                    if 0 <= n_cell[0] < map_w and 0 <= n_cell[1] < map_h: #if the cell is within bounds
                        if self.map["array"][n_cell[1]][n_cell[0]] == 0:
                            neighbors.append(n_cell)
            
            return neighbors
        
        if visualize:
            self.clear_points()

        while queue:
            _, curr_path_cost, curr_path = heapq.heappop(queue)
            curr_cell = curr_path[-1]

            if curr_cell in seen:
                continue

            seen.add(curr_cell)
            if visualize and len(curr_path) >= 2:
                prev_cell = curr_path[-2]
                prev_point, curr_point = self.grid_to_real_frame([prev_cell, curr_cell])
                
                edges.append((prev_point, curr_point))
                # Publish incrementally to show tree expansion over time
                self.publish_edges(edges)

            if curr_cell == end_cell:
                return np.array(curr_path)

            for neighbor in find_valid_neighbors(curr_cell):
                new_path = curr_path + (neighbor,)
                new_path_cost = curr_path_cost + math.dist(curr_cell, neighbor)
                new_total_cost = new_path_cost + math.dist(neighbor ,end_cell)
                heapq.heappush(queue, (new_total_cost, new_path_cost, new_path))

    

    def publish_edges(self, edges):
        """Publish all accumulated edges as a single connected marker."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = "search_alg"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        # Add all edge points to the marker
        for start, end in edges:
            start_pt = Point()
            start_pt.x = start[0]
            start_pt.y = start[1]
            start_pt.z = 0.0

            end_pt = Point()
            end_pt.x = end[0]
            end_pt.y = end[1]
            end_pt.z = 0.0

            marker.points.extend([start_pt, end_pt])

        marker.pose.orientation.w = 1.0
        marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
        marker.scale.x = 0.02  # Line width in meters
        marker.scale.z = 0.0
        
        marker_arr = MarkerArray()
        marker_arr.markers.append(marker)
        self.search_pub.publish(marker_arr)
    
    def clear_points(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = "search_alg"
        marker.action = Marker.DELETEALL

        marker_arr = MarkerArray()
        marker_arr.markers.append(marker)
        self.search_pub.publish(marker_arr)

    def shorten_cell_path(self, cell_path):
        curr_heading = None
        new_cell_path = []

        for i in range(1, len(cell_path)):
            prev_cell, new_cell = cell_path[i-1], cell_path[i]

            new_heading = (new_cell[0] - prev_cell[0], new_cell[1] - prev_cell[1])
            if new_heading != curr_heading:
                new_cell_path.append(prev_cell)
                curr_heading = new_heading
        
        new_cell_path.append(cell_path[-1])
        return new_cell_path

    def grid_to_real_frame(self, cells):
        cells = np.array([[cx * self.map["res"], cy * self.map["res"],0,1] for cx, cy in cells])
        points = (self.map["transform"] @ cells.T).T

        return np.array([[px,py] for px,py,_,_ in points])
    
    def real_to_grid_frame(self, points):
        points = np.array([[px,py,0,1] for px, py in points])
        cells = ((self.map["transform_inv"] @ points.T).T)
        return np.array([[cx / self.map["res"], cy / self.map["res"]] for cx , cy, _, _ in cells], dtype = int)


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
