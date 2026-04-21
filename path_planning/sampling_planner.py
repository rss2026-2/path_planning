import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from path_planning.utils import LineTrajectory
from rclpy.node import Node
import numpy as np
import cv2
import pickle
import networkx as nx
from path_planning.offline_prm import PRM
from scipy.spatial.transform import Rotation as R
# from rclpy.qos import QoSProfile, DurabilityPolicy

class PathPlan(Node):

    def __init__(self):
        super().__init__("sampling_planner")

        # -- Declared parameters --
        self.declare_parameter('odom_topic', "/initialpose")

        #seperate trajectory publishers for path comparison
        self.declare_parameter('viz_namespace', "/planned_trajectory")
        self.declare_parameter("viz_traj_color", [1.0,1.0,1.0,1.0])
        self.declare_parameter('publish_path', True)
        self.declare_parameter("path_topic", "/trajectory/current")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.viz_namespace = self.get_parameter("viz_namespace").get_parameter_value().string_value
        self.viz_traj_color = self.get_parameter("viz_traj_color").get_parameter_value().double_array_value
        self.publish_path = self.get_parameter('publish_path').get_parameter_value().bool_value
        self.path_topic = self.get_parameter("path_topic").get_parameter_value().string_value


        self.start_point = None
        self.end_point = None
        self.occupancy_map = None

        # -- Publishers and subscribers --
        # Pose estimate subscriber
        if self.odom_topic == "/odom":
            self.pose_sub = self.create_subscription(Odometry, self.odom_topic, self.pose_cb, 10)
        elif self.odom_topic == "/initialpose":
            self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, self.odom_topic, self.pose_cb, 10)
         
         # Current trajectory publisher
        if self.publish_path:
            self.traj_pub = self.create_publisher(PoseArray, self.path_topic, 10)
        
        # Goal pose subscriber
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 10)
        
        # Directories
        map_path = f'src/path_planning/path_planning_prm/inflated_map.pkl'
        rm_path = f'src/path_planning/path_planning_prm/roadmap.pkl'
        rmtree_path = f'src/path_planning/path_planning_prm/roadmap_KDtree.pkl'

        self.get_logger().info("Attempting to load saved roadmap files...")

        # Load map package
        with open(map_path, 'rb') as f:
            map_package = pickle.load(f)
            self.occupancy_map = map_package['occupancy_map']
            self.resolution = map_package['resolution']
            self.origin_x = map_package['origin_x']
            self.origin_y = map_package['origin_y']
            self.map_yaw = map_package['map_yaw']
        
        # Load roadmap
        with open(rm_path, 'rb') as f:
            self.PRM_map = pickle.load(f)

        # Load roadmap tree
        with open(rmtree_path, 'rb') as f:
            self.tree = pickle.load(f)

        self.get_logger().info("Successfully loaded saved files!")
        # -- Initialized variables --
        self.start_point = None
        self.end_point = None
        self.trajectory = LineTrajectory(node=self, viz_namespace = self.viz_namespace)

        self.get_logger().info("Ready to start planning!")

    def link_node_to_graph(self, point, node_id, radius=5.0):
        """
        Adds node to a networkx graph. Uses KDTree to quickly find the closest neighbors in
        the graph to connect to the node.

        Args:
            point (tuple): (x_coord, y_coord)
            node_id (int/str): the id of the node
            radius (float): connects node to any nodes in the graph within this radius

        Returns:
            node_id (int/str): the id of the node
        """
        if self.tree is None:
            return node_id

        self.PRM_map.add_node(node_id, pos=point)
        neighbor_indices = self.tree.query_ball_point(point, r=radius)

        for idx in neighbor_indices:
            neighbor_id = list(self.tree.data)[idx]

            for n_id, data in self.PRM_map.nodes(data=True):
                if n_id != node_id and np.allclose(data['pos'], neighbor_id):
                    dist, clear = self.is_line_clear(point, data['pos'])
                    if clear:
                        self.PRM_map.add_edge(node_id, n_id, weight=dist)
        return node_id

    def euclidean(self,u, v):
        pu = self.PRM_map.nodes[u]['pos']
        pv = self.PRM_map.nodes[v]['pos']
        return ((pu[0] - pv[0])**2 + (pu[1] - pv[1])**2)**0.5

    def manhattan(self,u, v):
        pu = self.PRM_map.nodes[u]['pos']
        pv = self.PRM_map.nodes[v]['pos']
        return abs(pu[0] - pv[0]) + abs(pu[1] - pv[1])

    def is_line_clear(self, p1, p2):
        """
        Makes sure the shortest line formed between p1 and p2 does not intersect with an obstacle.

        Args:
            p1 (tuple): (x_coord, y_coord)
            p2 (tuple): (x_coord, y_coord)
        Returns:
            bool
        """
        dist = np.linalg.norm(np.array(p1) - np.array(p2))
        n_steps = max(2, int(dist / (self.resolution * 0.5)))
        x_vals = np.linspace(p1[0], p2[0], n_steps)
        y_vals = np.linspace(p1[1], p2[1], n_steps)

        tx = x_vals - self.origin_x
        ty = y_vals - self.origin_y

        cos_q, sin_q = np.cos(-self.map_yaw), np.sin(-self.map_yaw)
        lx = tx * cos_q - ty * sin_q
        ly = tx * sin_q + ty * cos_q

        ixs = np.clip((lx / self.resolution).astype(int), 0, self.occupancy_map.shape[1] - 1)
        iys = np.clip((ly / self.resolution).astype(int), 0, self.occupancy_map.shape[0] - 1)

        if np.any(self.occupancy_map[iys, ixs] == 0):
            return dist, False
        return dist, True

    def plan_path(self, start_point, end_point):
        """
        Plans a path between the start and end point. Uses precomputed PRM graph to discretize
        states. Then uses astar to plan shortest path through the points. Uses euclidean distance
        heuristic for astar. Publishes trajectory to traj_pub.

        Args:
            start_point (tuple): (x_coords, y_coords)
            end_point (tuple): (x_coords, y_coords)
        Returns:
            None
        """

        if self.PRM_map is None or len(self.PRM_map.nodes) == 0:
            self.get_logger().warn("PRM not loaded yet.")
            return

        self.link_node_to_graph(start_point, "start")
        self.link_node_to_graph(end_point, "end")

        try:
            self.get_logger().info("dijkstra's")
            node_path = nx.astar_path(self.PRM_map, "start", "end", weight='weight')
            points = [self.PRM_map.nodes[n_id]['pos'] for n_id in node_path]

            self.trajectory.clear()
            for p in points:
                self.trajectory.addPoint(p)

            if self.publish_path:
                self.traj_pub.publish(self.trajectory.toPoseArray())
                self.get_logger().info("Path successfully planned and published.")

            self.trajectory.publish_viz(traj_color = self.viz_traj_color)
            self.get_logger().info("Path Visualized!")
            

        except nx.NetworkXNoPath:
            self.get_logger().error("No path possible between start and end.")

        if self.PRM_map.has_node('start'):
            self.PRM_map.remove_node('start')
        if self.PRM_map.has_node('end'):
            self.PRM_map.remove_node('end')

    def pose_cb(self, msg):
        """
        Initializes a start point for the path planner. Caches it in the node.

        Args:
            msg (Odometry): ROS2 message type (topic is odometry if in sim, initial_pose if irl)
        Returns:
            None
        """
        self.start_point = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.odom_topic == "/initialpose":
            self.get_logger().info(f"start point set: {self.start_point}")

    def goal_cb(self, msg):
        """
        Initalizes an end point for the path planner and calls the path planner.

        Args:
            msg (PoseStamped): ROS2 message type
        Returns:
            None
        """
        if self.start_point is None:
            self.get_logger().warn("Cannot plan: Start point (Odometry) not yet received.")
            return

        self.end_point = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f"goal pose set: {self.end_point}")
        self.plan_path(self.start_point, self.end_point)

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
