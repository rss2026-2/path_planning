import networkx as nx
import pickle
import random
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

class PRM():
    def __init__(self, occupancy_map, msg, n):
        print("INITIALIZED OFFLINE PRM NODE")

        self.roadmap = nx.Graph()
        self.occupancy_map = occupancy_map

        # Map specs
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.height = msg.info.height
        self.width = msg.info.width

        # for translating the map to the right spot
        quat = [
            msg.info.origin.orientation.x,
            msg.info.origin.orientation.y,
            msg.info.origin.orientation.z,
            msg.info.origin.orientation.w
        ]
        self.map_yaw = R.from_quat(quat).as_euler('xyz')[2]
        self.max_attempts = 100000

        rm, rmtree = self.generate_prm_star(n, 5.0)
        with open('src/path_planning/path_planning_prm/roadmap.pkl', 'wb') as f:
            pickle.dump(rm, f)
        with open('src/path_planning/path_planning_prm/roadmap_KDtree.pkl','wb') as f:
            pickle.dump(rmtree, f)

        print("KDTree saved to roadmap_KDtree.pkl and graph saved to roadmap.pkl")

    def world_to_grid(self, point):
        """
        Translates world points to grid points (necessary for map to be translated to the right spot
        in Rviz). Takes meters into pixels.

        Args:
            point (tuple): (x_coords, y_coords)
        Returns:
            (ix, iy) (tuple): pixel coordinates
        """
        tx = point[0] - self.origin_x
        ty = point[1] - self.origin_y

        cos_q = np.cos(-self.map_yaw)
        sin_q = np.sin(-self.map_yaw)

        rotated_x = tx * cos_q - ty * sin_q
        rotated_y = tx * sin_q + ty * cos_q

        ix = int(rotated_x / self.resolution)
        iy = int(rotated_y / self.resolution)
        return ix, iy

    def grid_to_world(self, ix, iy):
        """
        Translates grid points to world points.

        Args:
            ix (float): the x-value pixel point
            iy (float): the y-value pixel point

        Returns:
            (tuple): (x_coords, y_coords) in meters
        """
        lx = ix * self.resolution
        ly = iy * self.resolution

        cos_q = np.cos(self.map_yaw)
        sin_q = np.sin(self.map_yaw)

        tx = lx * cos_q - ly * sin_q
        ty = lx * sin_q + ly * cos_q

        return (tx + self.origin_x, ty + self.origin_y)

    def is_point_free(self, point):
        """
        Checks if a point collides with anything.

        Args:
            point (tuple): (x_coords, y_coords)

        Returns:
            bool
        """
        ix, iy = self.world_to_grid(point)
        if 0 <= ix < self.width and 0 <= iy < self.height:
            return self.occupancy_map[iy, ix] == 1
        return False

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

        ixs = np.clip((lx / self.resolution).astype(int), 0, self.width - 1)
        iys = np.clip((ly / self.resolution).astype(int), 0, self.height - 1)

        if np.any(self.occupancy_map[iys, ixs] == 0):
            return dist, False
        return dist, True

    def sample_random_point(self):
        """
        Samples uniformly in grid space, then transforms to world space.
        Args:
            None
        Returns:
            (tuple): (x_coords, y_coords)
        """
        ix = random.randint(0, self.width - 1)
        iy = random.randint(0, self.height - 1)
        return self.grid_to_world(ix, iy)

    def generate_prm_star(self, num_samples, connection_radius):
        """
        Generates both a graph and a KDTree that represents the PRM state space. Automatically
        prunes for feasible set of edges and nodes.

        Args:
            num_samples (int): the total number of nodes in the graph (size)
            connection_radius (float): the longest lenght and edge can be
        Returns:
            tuple(neworkx.Graph, scipy.spatial.KDTree)
        """
        attempts = 0
        while self.roadmap.number_of_nodes() < num_samples and attempts < self.max_attempts:
            attempts += 1
            point = self.sample_random_point()
            if self.is_point_free(point):
                self.roadmap.add_node(self.roadmap.number_of_nodes(), pos=point)

        if attempts >= self.max_attempts:
            print("WARNING: PRM reached max attempts before fulfilling sample quota.")

        node_ids = list(self.roadmap.nodes())
        node_coords = [self.roadmap.nodes[idx]['pos'] for idx in node_ids]
        self.tree = KDTree(node_coords)

        for node_id in self.roadmap.nodes:
            node_coords = self.roadmap.nodes[node_id]['pos']
            neighbor_indices = self.tree.query_ball_point(node_coords, r=connection_radius)

            for i in neighbor_indices:
                neighbor_id = node_ids[i]
                if neighbor_id <= node_id: # Avoid self-loops and duplicate edges
                    continue

                neighbor_coords = self.roadmap.nodes[neighbor_id]['pos']
                dist, clear = self.is_line_clear(node_coords, neighbor_coords)
                if clear:
                    self.roadmap.add_edge(node_id, neighbor_id, weight=dist)

        return self.roadmap, self.tree
