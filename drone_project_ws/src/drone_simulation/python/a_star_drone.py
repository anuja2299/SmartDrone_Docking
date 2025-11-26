import numpy as np
import json
import time
import heapq
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt

class Node:
    def __init__(self, position, parent=None):
        self.position = np.array(position)  # [x, y, z]
        self.parent = parent
        self.g = 0  # Cost from start
        self.h = 0  # Heuristic cost to goal
        self.f = 0  # Total cost (g + h)

    def __lt__(self, other):
        return self.f < other.f

class AStarDrone:
    def __init__(self):
        self.drone_pos = np.array([0.0, 0.0, 0.0])  # Initial position from odometry
        self.target_pos = np.array([2.8, 0.2, 0.0])  # Target position
        self.grid_size = 5  # 20x20 grid (5 cm cells)
        self.actions = [
            np.array([1, 0, 0]),  # Right
            np.array([-1, 0, 0]),  # Left
            np.array([0, 1, 0]),  # Up
            np.array([0, -1, 0]),  # Down
            np.array([1, 1, 0]),   # Up-Right
            np.array([-1, 1, 0]),  # Up-Left
            np.array([1, -1, 0]),  # Down-Right
            np.array([-1, -1, 0])  # Down-Left
        ]  # 5 cm steps
        self.html_output = []
        self.path = []
        self.cmd_vel_pub = None
        self.odom_sub = None
        rclpy.init()
        self.node = rclpy.create_node('a_star_drone')
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.node.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.rate = self.node.create_rate(10)  # 10 Hz

    def odom_callback(self, msg):
        self.drone_pos = np.array([msg.pose.pose.position.x * 100,  # Convert to cm
                                   msg.pose.pose.position.y * 100,
                                   msg.pose.pose.position.z * 100])

    def get_pattern_position(self):
        for _ in range(3):
            try:
                with open("pattern_position.txt", "r") as f:
                    content = f.read().strip()
                    if not content:
                        self.custom_print("Warning: Pattern position file is empty. Using last known position.")
                        return self.target_pos
                    pos = json.loads(content)
                    self.target_pos = np.array([float(pos["x"]), float(pos["y"]), float(pos["z"])])
                    return self.target_pos
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                self.custom_print(f"Error reading pattern position: {e}. Retrying...")
                time.sleep(0.05)
        self.custom_print("Failed to read pattern position. Using last known position.")
        return self.target_pos

    def heuristic(self, pos):
        # Euclidean distance to target (ignoring z for 2D plane)
        return np.sqrt(np.sum((pos[:2] - self.target_pos[:2]) ** 2))

    def is_valid(self, pos):
        # Allow positions from 0 to 99 cm, mock obstacle check
        return all(0 <= p <= 99 for p in pos[:2]) and not self.is_obstacle(pos)

    def is_obstacle(self, pos):
        # Mock obstacle detection; replace with Gazebo sensor data
        return False  # Extend with real obstacle data (e.g., from point cloud)

    def get_neighbors(self, node):
        neighbors = []
        for action in self.actions:
            new_pos = node.position + action * self.grid_size  # 5 cm step
            if self.is_valid(new_pos):
                neighbors.append(Node(new_pos, node))
        return neighbors

    def a_star(self):
        start = Node(self.drone_pos)
        goal = Node(self.target_pos)
        open_set = []
        heapq.heappush(open_set, start)
        closed_set = set()
        self.custom_print(f"Starting A* from {start.position} to {goal.position}")

        while open_set:
            current = heapq.heappop(open_set)
            self.custom_print(f"Exploring node at {current.position}, f={current.f}")
            if np.all(np.abs(current.position - goal.position) < 1):  # Within 1 cm
                path = []
                while current:
                    path.append(current.position)
                    current = current.parent
                self.custom_print(f"Path found with length {len(path)} nodes")
                return path[::-1]  # Reverse to get start to goal

            closed_set.add(tuple(current.position))
            for neighbor in self.get_neighbors(current):
                if tuple(neighbor.position) in closed_set:
                    continue

                neighbor.g = current.g + self.grid_size  # Cost of 5 cm move
                neighbor.h = self.heuristic(neighbor.position)
                neighbor.f = neighbor.g + neighbor.h

                if any(n.position.tolist() == neighbor.position.tolist() and n.f <= neighbor.f for n in open_set):
                    continue

                heapq.heappush(open_set, neighbor)

        self.custom_print("No valid path found to target.")
        return []

    def custom_print(self, *args, **kwargs):
        message = " ".join(str(arg) for arg in args)
        self.html_output.append(message)
        print(message, **kwargs)

    def apply_path(self):
        self.custom_print("Starting A* pathfinding and drone movement...")
        path = self.a_star()
        if path:
            self.path = path
            for i, pos in enumerate(path):
                self.drone_pos = pos
                distance = np.linalg.norm(self.drone_pos[:2] - self.target_pos[:2])
                self.custom_print(f"Step {i+1}: Moved to {self.drone_pos}, Distance to Target: {distance:.2f} cm")
                action = (pos - self.drone_pos[:3]) / 50  # Velocity for 5 cm step (0.1 m/s)
                twist = Twist()
                twist.linear.x = action[0]  # m/s
                twist.linear.y = action[1]
                twist.linear.z = action[2]
                self.cmd_vel_pub.publish(twist)
                self.custom_print(f"Action Applied: Velocity ({action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}) m/s")
                self.rate.sleep()  # Sync with 10 Hz
                if distance < 1:
                    final_error = distance
                    self.custom_print(f"Reached target at Step {i+1} with position {self.drone_pos}, Final Error: {final_error:.2f} cm")
                    break
        else:
            self.custom_print("Pathfinding failed; no movement applied.")

    def save_outputs(self):
        with open("a_star_output.html", "w") as f:
            f.write("<html><head><style>body {font-family: Arial, sans-serif; background-color: #f0f0f0; color: #333;}</style></head><body><h2>A* Drone Movement Log</h2><pre>")
            for line in self.html_output:
                f.write(f"{line}<br>\n")
            f.write("</pre></body></html>")

def main():
    a_star = AStarDrone()
    a_star.get_pattern_position()  # Update target position
    a_star.apply_path()
    a_star.save_outputs()
    # Handle plotting safely with correct scale
    if a_star.path and len(a_star.path) > 0:
        path_2d = np.array(a_star.path)
        if path_2d.ndim == 1 and len(path_2d) == 3:
            path_2d = path_2d.reshape(1, 3)
        path_2d = path_2d[:, :2]  # Extract x, y coordinates
        plt.figure(figsize=(8, 6))
        plt.plot(path_2d[:, 0], path_2d[:, 1], 'b-', label='A* Path')
        plt.plot(a_star.drone_pos[0], a_star.drone_pos[1], 'ro', label='Start')
        plt.plot(a_star.target_pos[0], a_star.target_pos[1], 'g*', label='Target')
        plt.title('A* Path to Target')
        plt.xlabel('X Position (cm)')
        plt.ylabel('Y Position (cm)')
        plt.xlim(0, 100)  # Correct scale for 100x100 cm arena
        plt.ylim(0, 100)
        plt.grid(True)
        plt.legend()
        plt.savefig('a_star_path.png')
        plt.show()
    else:
        print("No path to plot due to failed pathfinding.")
    a_star.node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
