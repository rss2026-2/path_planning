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
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "/initialpose")
        self.declare_parameter('map_topic', "/map")
        self.declare_parameter('rover_radius', 0.60)
        self.declare_parameter('offline', False)
        self.declare_parameter('num_nodes', 250)

        self.rover_radius = self.get_parameter('rover_radius').value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.offline = self.get_parameter('offline').value
        self.num_nodes = self.get_parameter('num_nodes').value

        self.start_point = None
        self.end_point = None
        self.occupancy_map = None

        self.PRM_map = nx.Graph()
        self.tree = None

        with open("/root/racecar_ws/path_planning_prm/roadmap_KDtree_big.pkl", 'rb') as f:
            self.tree = pickle.load(f)

        with open("/root/racecar_ws/path_planning_prm/roadmap_big.pkl", 'rb') as f:
            self.PRM_map = pickle.load(f)

        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, 1)
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 10)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, self.odom_topic, self.pose_cb, 10)
        self.traj_pub = self.create_publisher(PoseArray, "/trajectory/current", 10)

        # use this publisher for when visualizing several planners in one go
        # latch_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        # self.traj_pub = self.create_publisher(PoseArray, "/trajectory/current", latch_qos)

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

    def map_cb(self, msg):
        """
        The callback for the offline PRM planner. Takes in information from the map ones
        and generates a graph of nodes in the map. Nodes are connected with edges if there is a linear
        ray between them that does not pass through an obstacle.

        Args:
            msg (OccupancyGrid) : ROS2 message that represents the map

        Returns:
            None
        """
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        quat = [msg.info.origin.orientation.x, msg.info.origin.orientation.y,
                msg.info.origin.orientation.z, msg.info.origin.orientation.w]
        self.map_yaw = R.from_quat(quat).as_euler('xyz')[2]
        pixel_radius = int(self.rover_radius / self.resolution)

        map_data = np.array(msg.data, np.double)
        binary_map = (map_data == 0).astype(np.uint8)

        dist_map = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5)
        safe_map = (dist_map > pixel_radius).astype(np.int8)

        height, width = msg.info.height, msg.info.width
        self.occupancy_map = safe_map.reshape((height, width))

        if self.offline:
            self.get_logger().info("Map received. Generating PRM...")
            self.get_logger().info(f"{self.resolution}")
            self.get_logger().info(f"{height}")
            self.get_logger().info(f"{width}")
            self.get_logger().info(f"{self.origin_x}")
            self.get_logger().info(f"""{msg.info.origin.orientation.x, msg.info.origin.orientation.y,
                    msg.info.origin.orientation.z, msg.info.origin.orientation.w}""")
            prm_generator = PRM(self.occupancy_map, msg, self.num_nodes)
            self.PRM_map = prm_generator.roadmap
            self.tree = prm_generator.tree
            self.get_logger().info("PRM ready for planning.")

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

            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()
            self.get_logger().info("Path successfully planned and published.")

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
