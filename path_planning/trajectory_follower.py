
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive # added AckermannDrive
import rclpy
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from .utils import LineTrajectory

# added:
from visualization_msgs.msg import Marker
import numpy as np
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as R

class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default") # /pf/pose/odom - the localization pf pose estimate
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        # self.lookahead = 0.8  # FILL IN # this was our default before
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

        # added in the last pure pursuit
        # self.declare_parameter("car_length", 0.325) # replaced with self.wheelbase_length
        self.declare_parameter("max_steering_angle", 0.34)
        self.declare_parameter("velocity", 1.0)
        self.declare_parameter("lookahead", 0.8)
        self.declare_parameter("error_epsilon", 1.0)
        self.declare_parameter("discretization_length", 1.0)
        # self.CAR_LENGTH = self.get_parameter('car_length').get_parameter_value().double_value # replaced with self.wheelbase_length
        self.MAX_STEERING_ANGLE = self.get_parameter('max_steering_angle').get_parameter_value().double_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value # replaced with self.speed
        self.LOOKAHEAD = self.get_parameter('lookahead').get_parameter_value().double_value
        self.EPSILON = self.get_parameter('error_epsilon').get_parameter_value().double_value
        self.STEERING_ANGLE_THRESH = 1.2 # initially working with it at 0.9 but it was reversing a lot

        self.DISCRETIZATION_LENGTH = self.get_parameter('discretization_length').get_parameter_value().double_value

        # this could mess things up:
        self.speed = self.VELOCITY
        # self.lookahead = self.LOOKAHEAD
        self.path = None
        self.line_pub = self.create_publisher(Marker, '/drive_line', 10)


    def pose_callback(self, odometry_msg):
        """
        Takes in a message of type Odometry which is our pose estimate from localization
        and caches the pose.
        """

        self.x = odometry_msg.pose.pose.position.x
        self.y = odometry_msg.pose.pose.position.y

        orientation = odometry_msg.pose.pose.orientation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        r = R.from_quat(quat)
        self.theta = r.as_euler('zxy', degrees=False)[0]
        self.get_logger().info(f'New Pose: {self.x}, {self.y}')

    def trajectory_callback(self, msg):
        """
        Callback function that runs when /trajectory/current is populated with a PoseArray.

        Generates a set of points discretizing the path (represented by key points) into points
        that can be followed using our implementation of pure pursuit.
        """
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True

        x, y = zip(*self.trajectory.points)
        VisualizationTools.plot_line(list(x), list(y), self.line_pub, color=(0.0, 1.0, 0.0))

        # added:  discretizing the path
        new_path = [self.trajectory.points[0]] # initialize with the first point
        new_distances = [0]
        for i in range(1, len(self.trajectory.distances)):
            cummulative_segment_length_to_p2 = self.trajectory.distances[i]
            p1, p2 = self.trajectory.points[i-1], self.trajectory.points[i]
            segment_length = cummulative_segment_length_to_p2 - self.trajectory.distances[i-1]
            if segment_length > self.DISCRETIZATION_LENGTH:
                extra_points = int (segment_length // self.DISCRETIZATION_LENGTH) - 1 # one point less than the number of segments
                new_x_pts = np.linspace(p1[0], p2[0], 2 + extra_points)
                new_y_pts = np.linspace(p1[1], p2[1], 2 + extra_points)
                new_segment_distance = segment_length / (extra_points + 1) # divide segment lengthh by the new number of segments i need
                next_distance = new_segment_distance + self.trajectory.distances[i-1] # starting from the last point
                for x_new, y_new in zip(new_x_pts[1:-1], new_y_pts[1:-1]): # skips p1 and p2
                    new_path.append((x_new, y_new))
                    new_distances.append(next_distance)
                    next_distance += new_segment_distance
            new_path.append(p2)
            new_distances.append(cummulative_segment_length_to_p2)

        self.path = np.array(new_path) # list of x, y tuples --> 2d array


        # set the x and y points for the end point (in the map frame)
        self.end_x, self.end_y = new_path[-1]

        # visualize the path
        # x, y = zip(*new_path)
        # VisualizationTools.plot_line(list(x), list(y), self.line_pub)

        self.get_logger().info(f'\n***New Path Recieved: {len(new_path)} points ***')

    def timer_callback(self):
        """
        Timer callback to generate new Drive command by pure pursuit using the cached pose.
        """
        # check that we only look to move when we have a trajectory and a pose estimate
        if not self.initialized_traj or self.x is None or self.y is None or self.theta is None:
            return
        drive_cmd = AckermannDriveStamped()
        # self.get_logger().info(f'New Drive Command')

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link' # is this still true?
        drive_cmd.header = header

        # Generate path -- this is now in the self.trajectory?
        # path = self.generate_hermite_path(self.relative_x, self.relative_y)
        # self.get_logger().info(f'Path: {path[0]}')
        # TODO: Decide how we turn given traj into a path we want to use
        path = self.path # this is turned into (num_points, 2) array in the path initialization

        # visualize path -- should now be visualized with the other visualization tools,
        # since the path is not dynamic, we should also do this once at the beginning.

        # goal_dist = np.sqrt(self.relative_x**2 + self.relative_y**2)


        # self.LOOKAHEAD = max(0.3, min(1.2, 0.5 * goal_dist))
        # self.get_logger().info(f'{self.LOOKAHEAD=}')

        # Get the lookahead target point (in map frame)
        target_point = self.get_lookahead_point(self.path)

        # Use the target point to update the drive command using our implementation of pure pursuit
        pure_pursuit_drive_cmd = self.update_control(target_point)

        # Update the command msg
        drive_cmd.drive = pure_pursuit_drive_cmd
        #################################

        # publish the drive command instead of saving it
        # self.drive_cmd = drive_cmd
        self.get_logger().info(f'Drive command sent: {drive_cmd.drive.speed}')
        self.drive_pub.publish(drive_cmd)

    def get_lookahead_point(self, path):
        """
        Returns the first point on the path at least LOOKAHEAD distance away.
        """

        # Put robot xy position into a numpy array
        robot_pos = np.array([self.x, self.y])

        # Calculate the squared distance between robot position and each point on the path
        dists = np.sum((path - robot_pos)**2, axis=1)

        # Get the index of the closest point
        closest_idx = np.argmin(dists)

        # Only consider values further along the path than the closest point
        future_points = path[closest_idx:]
        future_dists = dists[closest_idx:]


        # Make a mask that is all the squared distances greater than the lookahead distance (squared)
        valid_mask = future_dists >= self.LOOKAHEAD**2

        # Apply the mask to our points
        valid_points = future_points[valid_mask]

        # If there is at least one valid point,
        if len(valid_points) > 0:
            # Return the first point in the array. This is the point closest to the lookahead distance
            return np.array(list(valid_points[0]))
        # If there are no valid points,
        else:
            # Just return the last point in the path as a fallback
            self.get_logger().info(f"No valid points > LOOKAHEAD to follow. Following last point")
            return np.array(list(path[-1]))

    def update_control(self, target_point):
        """
        Returns the ackerman drive command
        """
        drive = AckermannDrive()

        # -- Don't worry about reversing for now --
        # # in the case that the cone is behind the car, can also be modified for when we don't see the car
        # if self.relative_x < 0:
        #     drive.speed = -0.5
        #     # steer toward the cone while reversing
        #     drive.steering_angle = float(np.clip(
        #         -np.sign(self.relative_y) * self.MAX_STEERING_ANGLE * 0.6,
        #         -self.MAX_STEERING_ANGLE,
        #         self.MAX_STEERING_ANGLE
        #     ))
        #     return drive


        # Check to see if we are too close
        # TODO: make relative_x and relative_y be the end point in the base_link frame (done in path initialization)
        # TODO: this should calculate the distance from self.x and self.y to the end point
        goal_dist = np.sqrt((self.end_x - self.x)**2 + (self.end_y - self.y)**2)

        # if we are in the stopping range and pointed at the cone, it's okay
        # if goal_dist < self.parking_distance_max land goal_dist > self.parking_distance_min and self.pointed_at_cone():
        if goal_dist < self.EPSILON:
            drive.speed = 0.0
            drive.steering_angle = 0.0
            return drive

        # calculate with the pure pursuit
        robot_pos = np.array([self.x, self.y])
        # goal_vector = target_point - robot_pos
        goal_vector = self.world_to_vehicle(target_point)
        new_steering_angle = self.compute_feedback_angle(goal_vector)

        # If the turn we have to make is too tight or the cone is cut off, or the cone is just plainly too close, reverse first
        turning_angle_too_tight = abs(new_steering_angle) > self.MAX_STEERING_ANGLE * self.STEERING_ANGLE_THRESH
        cone_too_close = goal_dist < self.EPSILON
        if turning_angle_too_tight or cone_too_close:
            drive.speed = -0.5
            reverse_angle = -0.5 * new_steering_angle
            drive.steering_angle = float(np.clip(reverse_angle,
                                    -self.MAX_STEERING_ANGLE,
                                    self.MAX_STEERING_ANGLE))

        else: # if it is in front of us reasonable angle, give it that angle
            drive.steering_angle = float(np.clip(new_steering_angle,
                                    -self.MAX_STEERING_ANGLE,
                                    self.MAX_STEERING_ANGLE))

            drive.speed = self.get_speed_by_proximity(goal_dist)

        return drive

    def compute_feedback_angle(self, goal_vector):
        """
        Calculate steering angle by pure pursuit steering law
        """
        lookahead_dist = np.linalg.norm(goal_vector)

        # pure pursuit steering law
        delta = np.arctan2(
            2 * self.wheelbase_length * goal_vector[1],
            lookahead_dist**2
        )

        # delta = np.clip(delta, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        # not clipping because we want to check potential reverse behavior
        return delta

    def get_speed_by_proximity(self, distance_to_goal):
        """
        Return speed based on how close it is to the goal
        """
        if distance_to_goal > 1.25:
            return self.VELOCITY
        else:
            return 0.5

    def world_to_vehicle(self, point):
        dx = point[0] - self.x
        dy = point[1] - self.y

        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        x_car =  cos_theta * dx + sin_theta * dy
        y_car = -sin_theta * dx + cos_theta * dy

        return np.array([x_car, y_car])


class VisualizationTools:

    @staticmethod
    def plot_line(x, y, publisher, color=(1.0, 0.0, 0.0), frame="/map"):
        """
        Publishes the points (x, y) to publisher
        so they can be visualized in rviz as
        connected line segments.
        Args:
            x, y: The x and y values. These arrays
            must be of the same length.
            publisher: the publisher to publish to. The
            publisher must be of type Marker from the
            visualization_msgs.msg class.
            color: the RGB color of the plot.
            frame: the transformation frame to plot in.
        """
        # Construct a line
        line_strip = Marker()
        line_strip.type = Marker.POINTS  # LINE_STRIP for with lines
        line_strip.header.frame_id = frame

        # Set the size and color
        line_strip.scale.x = 0.1
        line_strip.scale.y = 0.1
        line_strip.color.a = 1.0
        line_strip.color.r = color[0]
        line_strip.color.g = color[1]
        line_strip.color.b = color[2]

        # Fill the line with the desired values
        for xi, yi in zip(x, y):
            p = Point()
            p.x = xi
            p.y = yi
            line_strip.points.append(p)

        # Publish the line
        publisher.publish(line_strip)



def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
