from geometry_msgs.msg import PoseArray, Pose
from rclpy.node import Node
import pickle
import rclpy

class RoadmapVisualizer(Node):
    def __init__(self):
        super().__init__('roadmap_viz')
        self.publisher = self.create_publisher(PoseArray, '/roadmap_nodes', 10)
        with open("roadmap.pkl", 'rb') as f:
            roadmap = pickle.load(f)
        print(f"Successfully loaded roadmap with {roadmap.number_of_nodes()} nodes.")
        self.roadmap = roadmap
        self.timer = self.create_timer(1.0, self.publish_roadmap)

    def publish_roadmap(self):
        msg = PoseArray()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        for node_id, data in self.roadmap.nodes(data=True):
            pose = Pose()
            pose.position.x = float(data['pos'][0])
            pose.position.y = float(data['pos'][1])
            pose.position.z = 0.0
            # Quaternion for "no rotation"
            pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.get_logger().info("published")
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    vis = RoadmapVisualizer()
    rclpy.spin(vis)
    rclpy.shutdown()
