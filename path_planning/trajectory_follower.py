import rclpy

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from .utils import LineTrajectory


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default") # /pf/pose/odom - the localization pf pose estimate
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 0.8  # FILL IN # this was our default before
        self.speed = 1.0  # FILL IN # we want to test with different speeds
        self.wheelbase_length = 0.325 # FILL IN # Need to check this number

        self.initialized_traj = False
        self.trajectory = LineTrajectory(self, "/followed_trajectory")

        self.pose_sub = self.create_subscription(Odometry,
                                                 self.odom_topic,
                                                 self.pose_callback,
                                                 1)
        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1) # publish drive commands here

        # Added
        self.x = None
        self.y = None
        self.theta = None

        timer_rate = 25
        self.create_timer(1/timer_rate, self.timer_callback)

        # added in the last pure persuit
        # self.declare_parameter("car_length", 0.325) # replaced with self.wheelbase_length
        self.declare_parameter("max_steering_angle", 0.34)
        self.declare_parameter("velocity", 1.0)
        self.declare_parameter("lookahead", 0.8)
        self.declare_parameter("error_epsilon", 0.05)
        self.declare_parameter("detection_mode", "cone")
        # self.CAR_LENGTH = self.get_parameter('car_length').get_parameter_value().double_value # replaced with self.wheelbase_length
        self.MAX_STEERING_ANGLE = self.get_parameter('max_steering_angle').get_parameter_value().double_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value # replaced with self.speed
        self.LOOKAHEAD = self.get_parameter('lookahead').get_parameter_value().double_value
        self.EPSILON = self.get_parameter('error_epsilon').get_parameter_value().double_value
        self.STEERING_ANGLE_THRESH = 1.2 # initially working with it at 0.9 but it was reversing a lot

        self.DETECTION_MODE = self.get_parameter('detection_mode').get_parameter_value().string_value

        # this could mess things up:
        self.speed = self.VELOCITY
        self.lookahead = self.LOOKAHEAD




    def pose_callback(self, odometry_msg):
        """
        Takes in a message of type Odometry which is our pose estimate from localization
        and caches the pose.
        """
        odometry_msg.header.stamp = self.get_clock().now().to_msg()
        odometry_msg.header.frame_id = "map"
        odometry_msg.child_frame_id = self.particle_filter_frame

        self.x = odometry_msg.pose.pose.position.x
        self.y = odometry_msg.pose.pose.position.y

        orientation = odometry_msg.pose.pose.orientation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        r = R.from_quat(quat)
        self.theta = r.as_euler('z', degrees=False)[0]

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True

    def timer_callback(self):
        """
        Timer callback to generate new Drive command by pure persuit using the cached pose.
        """
        # check that we only look to move when we have a trajectory and a pose estimate
        if not self.initialized_traj or self.x is None or self.y is None or self.theta is None:
            return
        drive_cmd = AckermannDriveStamped()
        # self.get_logger().info(f'New Drive Command')

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link'
        drive_cmd.header = header

        # Generate path -- this is now in the self.trajectory?
        # path = self.generate_hermite_path(self.relative_x, self.relative_y)
        # self.get_logger().info(f'Path: {path[0]}')
        # TODO: Decide how we turn given traj into a path we want to use

        # visualize path -- should now be visualized with the other visualization tools
        # x, y = zip(*path)
        # VisualizationTools.plot_line(list(x), list(y), self.line_pub)
        # goal_dist = np.sqrt(self.relative_x**2 + self.relative_y**2)

        # self.LOOKAHEAD = max(0.3, min(1.2, 0.5 * goal_dist))
        # self.get_logger().info(f'{self.LOOKAHEAD=}')

        # choose lookahead target  -- they gave us some math we can use for this
        target_point = self.get_lookahead_point(path)

        pure_persuit_drive_cmd = self.update_control(target_point)

        drive_cmd.drive = pure_persuit_drive_cmd
        #################################

        # publish the drive command instead of saving it
        # self.drive_cmd = drive_cmd
        self.drive_pub.publish(drive_cmd)

    def get_lookahead_point(self, path):
        """
        Returns the first point on the path at least LOOKAHEAD distance away.
        """
        # for p in path:
        #     if np.linalg.norm(p) > self.LOOKAHEAD:
        #         return p

        # return path[-1]
        dists = np.linalg.norm(path, axis=1)
        closest_idx = np.argmin(dists)

        for i in range(closest_idx, len(path)):
            if np.linalg.norm(path[i]) > self.LOOKAHEAD:
                return path[i]

        return path[-1]

    def update_control(self, target_point):
        """
        Returns the ackerman drive command
        """
        drive = AckermannDrive()
        # in the case that the cone is behind the car, can also be modified for when we don't see the car
        if self.relative_x < 0:
            drive.speed = -0.5
            # steer toward the cone while reversing
            drive.steering_angle = float(np.clip(
                -np.sign(self.relative_y) * self.MAX_STEERING_ANGLE * 0.6,
                -self.MAX_STEERING_ANGLE,
                self.MAX_STEERING_ANGLE
            ))
            return drive


        # Check to see if we are too close
        goal_dist = np.sqrt(self.relative_x**2 + self.relative_y**2)

        # if we are in the stopping range and pointed at the cone, it's okay
        if goal_dist < self.parking_distance_max and goal_dist > self.parking_distance_min and self.pointed_at_cone():
            # TODO add something here if we are too close but not pointed well...
            drive.speed = 0.0
            drive.steering_angle = 0.0
            return drive

        # calculate with the pure persuit
        new_steering_angle = self.compute_feedback_angle(target_point)

        # If the turn we have to make is too tight or the cone is cut off, or the cone is just plainly too close, reverse first
        turning_angle_too_tight = abs(new_steering_angle) > self.MAX_STEERING_ANGLE * self.STEERING_ANGLE_THRESH
        detected_cone_too_close = False #self.proximity_check
        cone_too_close = goal_dist < self.parking_distance_min
        if turning_angle_too_tight or  detected_cone_too_close or cone_too_close:
            drive.speed = -0.5
            reverse_angle = -0.5 * new_steering_angle
            drive.steering_angle = float(np.clip(reverse_angle,
                                    -self.MAX_STEERING_ANGLE,
                                    self.MAX_STEERING_ANGLE))

        else: # if it is in front of us reasonable angle, give it that angle
            drive.steering_angle = float(np.clip(new_steering_angle,
                                    -self.MAX_STEERING_ANGLE,
                                    self.MAX_STEERING_ANGLE))

            drive.speed = self.get_speed_by_mode_and_proximity(goal_dist)

        return drive

    def compute_feedback_angle(self, target_point):
        """
        Calculate steering angle?
        """
        lookahead_dist = np.linalg.norm(target_point)

        # pure pursuit steering law
        delta = np.arctan2(
            2 * self.wheelbase_length * target_point[1],
            lookahead_dist**2
        )

        # delta = np.clip(delta, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        # not clipping because we want to check potential reverse behavior
        return delta
    def get_speed_by_mode_and_proximity(self, distance_to_obj):
        # for the cone, we want to slow down as we appraoch the cone
        if self.DETECTION_MODE == "cone":
            # slow down near goal
            if distance_to_obj > 1.25:
                return self.VELOCITY
            else:
                return 0.5
        # for the line detection mode, don't want to scale speed because we want to go full speed always
        else:
            return self.VELOCITY

def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
