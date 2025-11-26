import gym
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import time

class DroneEnv(gym.Env, Node):
    def __init__(self):
        # Initialize the ROS2 node
        super().__init__('drone_env_node')
        gym.Env.__init__(self)

        # Define action and observation spaces
        # Actions: linear velocities in x, y, z (continuous)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # Observations: [drone_x, drone_y, drone_z, target_x, target_y, target_z]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Drone's current position
        self.drone_position = np.zeros(3, dtype=np.float32)
        # Target position from pattern_detector
        self.target_position = np.zeros(3, dtype=np.float32)

        # ROS2 Subscribers and Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.pattern_sub = self.create_subscription(PointStamped, '/pattern_position', self.pattern_callback, 10)

        # Wait for initial messages
        self.get_logger().info("Waiting for initial odometry and pattern position...")
        while not hasattr(self, 'drone_position_initialized') or not hasattr(self, 'target_position_initialized'):
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)

    def odom_callback(self, msg):
        # Update drone's position from odometry
        self.drone_position[0] = msg.pose.pose.position.x
        self.drone_position[1] = msg.pose.pose.position.y
        self.drone_position[2] = msg.pose.pose.position.z
        self.drone_position_initialized = True

    def pattern_callback(self, msg):
        # Convert pattern_detector's normalized image coordinates to world coordinates
        # Camera is at (0, 0, 1.45), FOV radius on ground is 0.367m
        # Normalized coordinates (-1 to 1) map to -0.367 to 0.367m
        norm_x = msg.point.x  # e.g., 0.545
        norm_y = msg.point.y  # e.g., 0.545
        world_x = norm_x * 0.367  # e.g., 0.545 * 0.367 ≈ 0.2
        world_y = norm_y * 0.367  # e.g., 0.545 * 0.367 ≈ 0.2
        world_z = 0.01  # QR code is at z=0.01
        self.target_position = np.array([world_x, world_y, world_z], dtype=np.float32)
        self.target_position_initialized = True

    def reset(self):
        # Reset the environment
        # Stop the drone
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        self.cmd_vel_pub.publish(cmd)

        # Wait for updated position
        time.sleep(0.1)
        rclpy.spin_once(self, timeout_sec=0.1)

        # Return the initial observation
        return self._get_observation()

    def step(self, action):
        # Apply the action (velocity commands)
        cmd = Twist()
        cmd.linear.x = float(action[0])  # Velocity in x
        cmd.linear.y = float(action[1])  # Velocity in y
        cmd.linear.z = float(action[2])  # Velocity in z
        self.cmd_vel_pub.publish(cmd)

        # Wait for the action to take effect
        time.sleep(0.1)
        rclpy.spin_once(self, timeout_sec=0.1)

        # Get the new observation
        obs = self._get_observation()

        # Compute the reward
        distance = np.linalg.norm(self.drone_position - self.target_position)
        reward = -distance  # Negative distance to encourage getting closer
        if distance < 0.05:  # If within 5cm of the target
            reward += 100.0  # Bonus for reaching the target

        # Check if done
        done = distance < 0.05

        # Additional info
        info = {'distance': distance}

        return obs, reward, done, info

    def _get_observation(self):
        # Observation: [drone_x, drone_y, drone_z, target_x, target_y, target_z]
        return np.concatenate([self.drone_position, self.target_position])

    def close(self):
        # Stop the drone and clean up
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        self.cmd_vel_pub.publish(cmd)
        self.destroy_node()

def main():
    rclpy.init()
    env = DroneEnv()
    env.reset()
    env.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
