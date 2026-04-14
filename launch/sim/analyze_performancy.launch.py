from launch import LaunchDescription
from launch_ros.actions import Node

sample_sizes = [5000]

def generate_launch_description():
    nodes = []

    for n in sample_sizes:
        nodes.append(
            Node(
                package='path_planning',
                executable='trajectory_planner',
                name=f'trajectory_planner_n{n}',
                parameters=[{
                    'offline': False,
                    'num_nodes': n,
                    'rover_radius': 0.20,
                }],
                remappings=[
                    ('/trajectory/current', f'/trajectory/n{n}'),
                ],
            )
        )

    nodes.append(
        Node(
            package='path_planning',
                executable='analyze_prm',
                name=f'analyze_prm',
        )
    )

    return LaunchDescription(nodes)
