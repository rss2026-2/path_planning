from geometry_msgs.msg import PoseArray, Pose
from rclpy.node import Node
from visualization_msgs.msg import Marker

import pickle
import rclpy

from viz_utils.visualization_tools import VisualizationTools

class RoadmapVisualizer(Node):
    def __init__(self):
        super().__init__('roadmap_viz')
        self.publisher = self.create_publisher(Marker, '/roadmap_nodes', 10)
        with open("src/path_planning/path_planning_prm/roadmap_0.30rad.pkl", 'rb') as f:
            roadmap = pickle.load(f)
        print(f"Successfully loaded roadmap with {roadmap.number_of_nodes()} nodes.")
        self.roadmap = roadmap
        self.timer = self.create_timer(1.0, self.publish_roadmap)

    def publish_roadmap(self):
        """
        Publish PRM as sphere markers in rviz.
        """

        stamp = self.get_clock().now().to_msg()
        frame = 'map'
        scale = (0.15, 0.15)
        color = (1.0, 0.0, 1.0)

        x_arr, y_arr = [], []

        for node_id, data in self.roadmap.nodes(data=True):
            x = float(data['pos'][0])
            y = float(data['pos'][1])

            x_arr.append(x)
            y_arr.append(y)
        
        VisualizationTools.draw_points(x_arr, y_arr, self.publisher, stamp, frame, type=Marker.POINTS, scale=scale, color=color)

        self.get_logger().info("published")

def main(args=None):
    rclpy.init(args=args)
    vis = RoadmapVisualizer()
    rclpy.spin(vis)
    rclpy.shutdown()
