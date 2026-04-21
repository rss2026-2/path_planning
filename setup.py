import glob
import os
from setuptools import find_packages
from setuptools import setup

package_name = 'path_planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/path_planning/launch/sim', glob.glob(os.path.join('launch', 'sim', '*launch.*'))),
        ('share/path_planning/launch/real', glob.glob(os.path.join('launch', 'real', '*launch.*'))),
        ('share/path_planning/launch/debug', glob.glob(os.path.join('launch', 'debug', '*launch.*'))),
        (os.path.join('share', package_name, 'config', 'sim'), glob.glob('config/sim/*.yaml')),
        (os.path.join('share', package_name, 'config', 'real'), glob.glob('config/real/*.yaml')),
        (os.path.join('share', package_name, 'config', 'debug'), glob.glob('config/debug/*.yaml')),
        ('share/path_planning/example_trajectories', glob.glob(os.path.join('example_trajectories', '*.traj'))),
        (os.path.join('share', package_name, 'maps'), glob.glob('maps/*.yaml')),
        (os.path.join('share', package_name, 'maps'), glob.glob('maps/*.png'))],
        
        
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sebastian',
    maintainer_email='sebastianag2002@gmail.com',
    description='Path Planning ROS2 Package',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory_builder = path_planning.trajectory_builder:main',
            'trajectory_loader = path_planning.trajectory_loader:main',
<<<<<<< HEAD
=======
            'trajectory_generator = path_planning.trajectory_generator:main',
            'trajectory_planner = path_planning.trajectory_planner:main',
>>>>>>> 7bae3d34fac47e0132ce378dd6d28cd0b5b5d5af
            'trajectory_follower = path_planning.trajectory_follower:main',
            'prm_viz = path_planning.PRM_visualizer:main',
            'analyze_prm = path_planning.analyze_prm:main',
            'grid_search_planner = path_planning.grid_search_planner:main',
            'sampling_planner = path_planning.sampling_planner:main',
            'path_analyzer = path_planning.analyze_plans:main'
        ],
    },
)
