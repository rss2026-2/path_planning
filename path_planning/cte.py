import numpy as np
import matplotlib.pyplot as plt
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import glob
import os

# -----------------------------
# USER CONFIG: time windows (seconds)
# -----------------------------
start_times = [9.5,37,25]
end_times   = [-1, -1, -1]   # -1 = run until end


# -----------------------------
# Compute CTE
# -----------------------------
def compute_cte(robot_pos, path):
    robot_pos = np.array(robot_pos)
    min_dist = float("inf")

    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i + 1])

        seg = p2 - p1
        seg_len2 = np.dot(seg, seg)

        if seg_len2 == 0:
            proj = p1
        else:
            t = np.dot(robot_pos - p1, seg) / seg_len2
            t = np.clip(t, 0.0, 1.0)
            proj = p1 + t * seg

        dist = np.linalg.norm(robot_pos - proj)
        min_dist = min(min_dist, dist)

    return min_dist


# -----------------------------
# Read ROS2 bag
# -----------------------------
def read_bag(bag_path):

    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id="sqlite3"
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    poses = []
    trajectory = None

    while reader.has_next():
        topic, data, t = reader.read_next()

        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)

        # -------------------------
        # robot pose
        # -------------------------
        if topic == "/pf/pose/odom":   # change to /pf/pose/odom if needed
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            poses.append((t, np.array([x, y])))

        # -------------------------
        # trajectory (PoseArray)
        # -------------------------
        elif topic == "/trajectory/current":
            trajectory = [
                (p.position.x, p.position.y)
                for p in msg.poses
            ]

    print("trajectory length:", 0 if trajectory is None else len(trajectory))
    print("poses:", len(poses))

    return poses, np.array(trajectory) if trajectory is not None else None


# -----------------------------
# Run analysis
# -----------------------------
bag_paths = sorted(glob.glob("./lab6_rosbags/v2/v1.5_run_*"))
print("Bag paths:", bag_paths)

plt.figure()

for i, bag in enumerate(bag_paths):
    print(f"\nProcessing {bag}")

    poses, traj = read_bag(bag)

    if traj is None or len(poses) == 0:
        print(f"Skipping {bag} (missing data)")
        continue

    # -----------------------------
    # TIME FILTERING (IMPORTANT FIX)
    # -----------------------------
    t0 = poses[0][0]
    filtered_poses = []

    for t, pos in poses:
        t_sec = (t - t0) * 1e-9

        if t_sec < start_times[i]:
            continue
        if end_times[i] != -1 and t_sec > end_times[i]:
            break

        filtered_poses.append((t, pos))

    poses = filtered_poses

    if len(poses) == 0:
        print(f"Skipping {bag} after filtering (no poses in time window)")
        continue

    # -----------------------------
    # Compute CTE vs time
    # -----------------------------
    times = []
    cte_values = []

    t0 = poses[0][0]

    for t, pos in poses:
        cte = compute_cte(pos, traj)
        times.append((t - t0) * 1e-9)
        cte_values.append(cte)

    plt.plot(times, cte_values, label=f"run {i+1}")


# -----------------------------
# Plot formatting
# -----------------------------
plt.xlabel("Time (s)")
plt.ylabel("Cross-Track Error (m)")
plt.title("CTE vs Time for 1.5m/s on car")
plt.grid(True)
plt.legend()

# -----------------------------
# Save figure
# -----------------------------
os.makedirs("results", exist_ok=True)
out_path = "results/cte_vs_time_v1.5.png"

plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"\nSaved to {out_path}")

plt.show()


# 1m/s on car
# import numpy as np
# import matplotlib.pyplot as plt
# import rosbag2_py
# from rclpy.serialization import deserialize_message
# from rosidl_runtime_py.utilities import get_message
# import glob
# import os

# # -----------------------------
# # USER CONFIG: time windows (seconds)
# # -----------------------------
# start_times = [0,0,17]
# end_times   = [-1, -1, -1]   # -1 = run until end


# # -----------------------------
# # Compute CTE
# # -----------------------------
# def compute_cte(robot_pos, path):
#     robot_pos = np.array(robot_pos)
#     min_dist = float("inf")

#     for i in range(len(path) - 1):
#         p1 = np.array(path[i])
#         p2 = np.array(path[i + 1])

#         seg = p2 - p1
#         seg_len2 = np.dot(seg, seg)

#         if seg_len2 == 0:
#             proj = p1
#         else:
#             t = np.dot(robot_pos - p1, seg) / seg_len2
#             t = np.clip(t, 0.0, 1.0)
#             proj = p1 + t * seg

#         dist = np.linalg.norm(robot_pos - proj)
#         min_dist = min(min_dist, dist)

#     return min_dist


# # -----------------------------
# # Read ROS2 bag
# # -----------------------------
# def read_bag(bag_path):

#     storage_options = rosbag2_py.StorageOptions(
#         uri=bag_path,
#         storage_id="sqlite3"
#     )

#     converter_options = rosbag2_py.ConverterOptions(
#         input_serialization_format="cdr",
#         output_serialization_format="cdr"
#     )

#     reader = rosbag2_py.SequentialReader()
#     reader.open(storage_options, converter_options)

#     topic_types = reader.get_all_topics_and_types()
#     type_map = {t.name: t.type for t in topic_types}

#     poses = []
#     trajectory = None

#     while reader.has_next():
#         topic, data, t = reader.read_next()

#         msg_type = get_message(type_map[topic])
#         msg = deserialize_message(data, msg_type)

#         # -------------------------
#         # robot pose
#         # -------------------------
#         if topic == "/pf/pose/odom":   # change to /pf/pose/odom if needed
#             x = msg.pose.pose.position.x
#             y = msg.pose.pose.position.y
#             poses.append((t, np.array([x, y])))

#         # -------------------------
#         # trajectory (PoseArray)
#         # -------------------------
#         elif topic == "/trajectory/current":
#             trajectory = [
#                 (p.position.x, p.position.y)
#                 for p in msg.poses
#             ]

#     print("trajectory length:", 0 if trajectory is None else len(trajectory))
#     print("poses:", len(poses))

#     return poses, np.array(trajectory) if trajectory is not None else None


# # -----------------------------
# # Run analysis
# # -----------------------------
# bag_paths = sorted(glob.glob("./lab6_rosbags/v1/run_*"))
# print("Bag paths:", bag_paths)

# plt.figure()

# for i, bag in enumerate(bag_paths):
#     print(f"\nProcessing {bag}")

#     poses, traj = read_bag(bag)

#     if traj is None or len(poses) == 0:
#         print(f"Skipping {bag} (missing data)")
#         continue

#     # -----------------------------
#     # TIME FILTERING (IMPORTANT FIX)
#     # -----------------------------
#     t0 = poses[0][0]
#     filtered_poses = []

#     for t, pos in poses:
#         t_sec = (t - t0) * 1e-9

#         if t_sec < start_times[i]:
#             continue
#         if end_times[i] != -1 and t_sec > end_times[i]:
#             break

#         filtered_poses.append((t, pos))

#     poses = filtered_poses

#     if len(poses) == 0:
#         print(f"Skipping {bag} after filtering (no poses in time window)")
#         continue

#     # -----------------------------
#     # Compute CTE vs time
#     # -----------------------------
#     times = []
#     cte_values = []

#     t0 = poses[0][0]

#     for t, pos in poses:
#         cte = compute_cte(pos, traj)
#         times.append((t - t0) * 1e-9)
#         cte_values.append(cte)

#     plt.plot(times, cte_values, label=f"run {i+1}")


# # -----------------------------
# # Plot formatting
# # -----------------------------
# plt.xlabel("Time (s)")
# plt.ylabel("Cross-Track Error (m)")
# plt.title("CTE vs Time for 1m/s on car")
# plt.grid(True)
# plt.legend()

# # -----------------------------
# # Save figure
# # -----------------------------
# os.makedirs("results", exist_ok=True)
# out_path = "results/cte_vs_time.png"

# plt.savefig(out_path, dpi=300, bbox_inches="tight")
# print(f"\nSaved to {out_path}")

# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# import rosbag2_py
# from rclpy.serialization import deserialize_message
# from rosidl_runtime_py.utilities import get_message
# import glob
# import os

# # -----------------------------
# # USER CONFIG: time windows (seconds)
# # -----------------------------
# start_times = [12, 47, 18]
# end_times   = [-1, -1, -1]   # -1 = run until end


# # -----------------------------
# # Compute CTE
# # -----------------------------
# def compute_cte(robot_pos, path):
#     robot_pos = np.array(robot_pos)
#     min_dist = float("inf")

#     for i in range(len(path) - 1):
#         p1 = np.array(path[i])
#         p2 = np.array(path[i + 1])

#         seg = p2 - p1
#         seg_len2 = np.dot(seg, seg)

#         if seg_len2 == 0:
#             proj = p1
#         else:
#             t = np.dot(robot_pos - p1, seg) / seg_len2
#             t = np.clip(t, 0.0, 1.0)
#             proj = p1 + t * seg

#         dist = np.linalg.norm(robot_pos - proj)
#         min_dist = min(min_dist, dist)

#     return min_dist


# # -----------------------------
# # Read ROS2 bag
# # -----------------------------
# def read_bag(bag_path):

#     storage_options = rosbag2_py.StorageOptions(
#         uri=bag_path,
#         storage_id="sqlite3"
#     )

#     converter_options = rosbag2_py.ConverterOptions(
#         input_serialization_format="cdr",
#         output_serialization_format="cdr"
#     )

#     reader = rosbag2_py.SequentialReader()
#     reader.open(storage_options, converter_options)

#     topic_types = reader.get_all_topics_and_types()
#     type_map = {t.name: t.type for t in topic_types}

#     poses = []
#     trajectory = None

#     while reader.has_next():
#         topic, data, t = reader.read_next()

#         msg_type = get_message(type_map[topic])
#         msg = deserialize_message(data, msg_type)

#         # -------------------------
#         # robot pose
#         # -------------------------
#         if topic == "/odom":   # change to /pf/pose/odom if needed
#             x = msg.pose.pose.position.x
#             y = msg.pose.pose.position.y
#             poses.append((t, np.array([x, y])))

#         # -------------------------
#         # trajectory (PoseArray)
#         # -------------------------
#         elif topic == "/trajectory/current":
#             trajectory = [
#                 (p.position.x, p.position.y)
#                 for p in msg.poses
#             ]

#     print("trajectory length:", 0 if trajectory is None else len(trajectory))
#     print("poses:", len(poses))

#     return poses, np.array(trajectory) if trajectory is not None else None


# # -----------------------------
# # Run analysis
# # -----------------------------
# bag_paths = sorted(glob.glob("./rosbabs/v2/run_*"))
# print("Bag paths:", bag_paths)

# plt.figure()

# for i, bag in enumerate(bag_paths):
#     print(f"\nProcessing {bag}")

#     poses, traj = read_bag(bag)

#     if traj is None or len(poses) == 0:
#         print(f"Skipping {bag} (missing data)")
#         continue

#     # -----------------------------
#     # TIME FILTERING (IMPORTANT FIX)
#     # -----------------------------
#     t0 = poses[0][0]
#     filtered_poses = []

#     for t, pos in poses:
#         t_sec = (t - t0) * 1e-9

#         if t_sec < start_times[i]:
#             continue
#         if end_times[i] != -1 and t_sec > end_times[i]:
#             break

#         filtered_poses.append((t, pos))

#     poses = filtered_poses

#     if len(poses) == 0:
#         print(f"Skipping {bag} after filtering (no poses in time window)")
#         continue

#     # -----------------------------
#     # Compute CTE vs time
#     # -----------------------------
#     times = []
#     cte_values = []

#     t0 = poses[0][0]

#     for t, pos in poses:
#         cte = compute_cte(pos, traj)
#         times.append((t - t0) * 1e-9)
#         cte_values.append(cte)

#     plt.plot(times, cte_values, label=f"run {i+1}")


# # -----------------------------
# # Plot formatting
# # -----------------------------
# plt.xlabel("Time (s)")
# plt.ylabel("Cross-Track Error (m)")
# plt.title("CTE vs Time for 1m/s in simulation")
# plt.grid(True)
# plt.legend()

# # -----------------------------
# # Save figure
# # -----------------------------
# os.makedirs("results", exist_ok=True)
# out_path = "results/cte_vs_time.png"

# plt.savefig(out_path, dpi=300, bbox_inches="tight")
# print(f"\nSaved to {out_path}")

# plt.show()


# SIMULATION 1 m/s
# import numpy as np
# import matplotlib.pyplot as plt
# import rosbag2_py
# from rclpy.serialization import deserialize_message
# from rosidl_runtime_py.utilities import get_message
# import glob
# import os

# # -----------------------------
# # USER CONFIG: time windows (seconds)
# # -----------------------------
# start_times = [12, 47, 18]
# end_times   = [-1, -1, -1]   # -1 = run until end


# # -----------------------------
# # Compute CTE
# # -----------------------------
# def compute_cte(robot_pos, path):
#     robot_pos = np.array(robot_pos)
#     min_dist = float("inf")

#     for i in range(len(path) - 1):
#         p1 = np.array(path[i])
#         p2 = np.array(path[i + 1])

#         seg = p2 - p1
#         seg_len2 = np.dot(seg, seg)

#         if seg_len2 == 0:
#             proj = p1
#         else:
#             t = np.dot(robot_pos - p1, seg) / seg_len2
#             t = np.clip(t, 0.0, 1.0)
#             proj = p1 + t * seg

#         dist = np.linalg.norm(robot_pos - proj)
#         min_dist = min(min_dist, dist)

#     return min_dist


# # -----------------------------
# # Read ROS2 bag
# # -----------------------------
# def read_bag(bag_path):

#     storage_options = rosbag2_py.StorageOptions(
#         uri=bag_path,
#         storage_id="sqlite3"
#     )

#     converter_options = rosbag2_py.ConverterOptions(
#         input_serialization_format="cdr",
#         output_serialization_format="cdr"
#     )

#     reader = rosbag2_py.SequentialReader()
#     reader.open(storage_options, converter_options)

#     topic_types = reader.get_all_topics_and_types()
#     type_map = {t.name: t.type for t in topic_types}

#     poses = []
#     trajectory = None

#     while reader.has_next():
#         topic, data, t = reader.read_next()

#         msg_type = get_message(type_map[topic])
#         msg = deserialize_message(data, msg_type)

#         # -------------------------
#         # robot pose
#         # -------------------------
#         if topic == "/odom":   # change to /pf/pose/odom if needed
#             x = msg.pose.pose.position.x
#             y = msg.pose.pose.position.y
#             poses.append((t, np.array([x, y])))

#         # -------------------------
#         # trajectory (PoseArray)
#         # -------------------------
#         elif topic == "/trajectory/current":
#             trajectory = [
#                 (p.position.x, p.position.y)
#                 for p in msg.poses
#             ]

#     print("trajectory length:", 0 if trajectory is None else len(trajectory))
#     print("poses:", len(poses))

#     return poses, np.array(trajectory) if trajectory is not None else None


# # -----------------------------
# # Run analysis
# # -----------------------------
# bag_paths = sorted(glob.glob("./rosbabs/run_*"))
# print("Bag paths:", bag_paths)

# plt.figure()

# for i, bag in enumerate(bag_paths):
#     print(f"\nProcessing {bag}")

#     poses, traj = read_bag(bag)

#     if traj is None or len(poses) == 0:
#         print(f"Skipping {bag} (missing data)")
#         continue

#     # -----------------------------
#     # TIME FILTERING (IMPORTANT FIX)
#     # -----------------------------
#     t0 = poses[0][0]
#     filtered_poses = []

#     for t, pos in poses:
#         t_sec = (t - t0) * 1e-9

#         if t_sec < start_times[i]:
#             continue
#         if end_times[i] != -1 and t_sec > end_times[i]:
#             break

#         filtered_poses.append((t, pos))

#     poses = filtered_poses

#     if len(poses) == 0:
#         print(f"Skipping {bag} after filtering (no poses in time window)")
#         continue

#     # -----------------------------
#     # Compute CTE vs time
#     # -----------------------------
#     times = []
#     cte_values = []

#     t0 = poses[0][0]

#     for t, pos in poses:
#         cte = compute_cte(pos, traj)
#         times.append((t - t0) * 1e-9)
#         cte_values.append(cte)

#     plt.plot(times, cte_values, label=f"run {i+1}")


# # -----------------------------
# # Plot formatting
# # -----------------------------
# plt.xlabel("Time (s)")
# plt.ylabel("Cross-Track Error (m)")
# plt.title("CTE vs Time for 1m/s")
# plt.grid(True)
# plt.legend()

# # -----------------------------
# # Save figure
# # -----------------------------
# os.makedirs("results", exist_ok=True)
# out_path = "results/cte_vs_time.png"

# plt.savefig(out_path, dpi=300, bbox_inches="tight")
# print(f"\nSaved to {out_path}")

# plt.show()
