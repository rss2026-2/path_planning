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

class GenerateTrajectory(Node):

    def __init__(self):
        super().__init__("trajectory_generator")

        # -- Declared parameters --
        self.declare_parameter('map_topic', "/map")
        self.declare_parameter('rover_radius', 0.30)
        self.declare_parameter('num_nodes', 250)
        self.declare_parameter('connection_radius', 5.0)
        self.declare_parameter('prm_label', "0.30rad")
        
        # Map topic
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        
        # PRM parameters
        self.rover_radius = self.get_parameter('rover_radius').value
        self.connection_radius = self.get_parameter('connection_radius').value
        self.num_nodes = self.get_parameter('num_nodes').value
        self.prm_label = self.get_parameter('prm_label').get_parameter_value().string_value
    

        # -- Publishers and subscribers -- 
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, 1)
        self.map_pub = self.create_publisher(OccupancyGrid, "/inflated_map", 10)

        # -- Initialized variables -- 
        self.tree = None
        self.occupancy_map = None

        self.PRM_map = nx.Graph()
        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")
        
        self.get_logger().info("=== GenerateTrajectory ready. Waiting for map to publish... === ")

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
        # Get map info attributes
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        height,width = msg.info.height, msg.info.width
        quat = [msg.info.origin.orientation.x, msg.info.origin.orientation.y,
                msg.info.origin.orientation.z, msg.info.origin.orientation.w]
        map_yaw = R.from_quat(quat).as_euler('xyz')[2]

        # Log that we're starting the graph generation
        self.get_logger().info("Map received. Generating PRM...")
        self.get_logger().info(f"""resolution: {resolution}; 
        height: {height}; width: {width}; origin: ({origin_x}, {origin_y}), 
        orientation: ({msg.info.origin.orientation.x, msg.info.origin.orientation.y, msg.info.origin.orientation.z, msg.info.origin.orientation.w})""")
        
        # Get the pixel radius from the connection radius and resolution
        pixel_radius = int(self.rover_radius / resolution)

        # Convert the map to a numpy array of map data
        map_arr = np.array(msg.data, np.double)
        # Convert the map to binary 
        binary_map = ((map_arr >= 0) & (map_arr < 50)).astype(np.uint8).reshape((height, width), order='C')  # reshape FIRST

        kernel = np.ones((3,3), np.uint8)
        binary_map = cv2.erode(binary_map, kernel)

        dist_map = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5)
        safe_map = (dist_map > pixel_radius).astype(np.int8)

        # Save map info to a dictionary package
        map_data = {
            'occupancy_map': safe_map,
            'resolution': resolution,
            'origin_x': origin_x,
            'origin_y': origin_y,
            'map_yaw': map_yaw
        }

        prm_generator = PRM(safe_map, msg, self.num_nodes, self.prm_label)
        rm, rmtree = prm_generator.generate_prm_star(self.num_nodes, self.connection_radius)

        # Initialize directories
        map_path = f'src/path_planning/path_planning_prm/inflated_map_{self.prm_label}.pkl'
        rm_path = f'src/path_planning/path_planning_prm/roadmap_{self.prm_label}.pkl'
        rmtree_path = f'src/path_planning/path_planning_prm/roadmap_KDtree_{self.prm_label}.pkl'

        # Save the package to disk
        with open(map_path, 'wb') as f:
            pickle.dump(map_data, f)
        
        # Save the roadmap to the disk
        with open(rm_path, 'wb') as f:
            pickle.dump(rm, f)

        # Save the roadmap tree to the disk
        with open(rmtree_path, 'wb') as f:
            pickle.dump(rmtree, f)

        self.get_logger().info(f"Map saved to {map_path}.\n KDTree saved to {rmtree_path}.\n Graph saved to {rm_path}")
        self.get_logger().info("PRM ready for planning.")

    def publish_map(self, msg):
        """
        Publishes the inflated map on the /inflated_map topic.

        Args:
            msg (ROS2 OccupancyGrid): the original map message

        Returns:
            None
        """
        inflated_map_msg = OccupancyGrid()
        inflated_map_msg.header = msg.header
        inflated_map_msg.info = msg.info

        original = np.array(msg.data, dtype=np.int16).reshape(
            (msg.info.height, msg.info.width)
        )

        inflated = original.copy()

        inflated[self.occupancy_map == 0] = 100

        inflated_only = (original == 0) & (self.occupancy_map == 0)
        inflated[inflated_only] = 50  

        inflated_map_msg.data = inflated.astype(np.int8).flatten().tolist()

        self.map_pub.publish(inflated_map_msg)
        self.get_logger().info("=== Published inflated map ===")

def main(args=None):
    rclpy.init(args=args)
    planner = GenerateTrajectory()
    rclpy.spin(planner)
    rclpy.shutdown()
