import rclpy
from rclpy.node import Node
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time

class TestDDPGNode(Node):
    def __init__(self):
        super().__init__('test_ddpg_node')
        
        # Initialize parameters
        self.state_dim = 6  # [drone_x, drone_y, drone_z, target_x, target_y, target_z]
        self.action_dim = 2  # [linear_x, linear_y]
        
        # Load the trained actor model from ddpg.py
        self.actor = models.Sequential([
            layers.Dense(64, activation='relu', input_dim=self.state_dim),
            layers.LayerNormalization(),
            layers.Dense(32, activation='relu'),
            layers.LayerNormalization(),
            layers.Dense(self.action_dim, activation='tanh')
        ])
        self.actor.load_weights('actor_weights.h5')  # Load saved weights
        self.get_logger().info("Loaded trained actor model from actor_weights.h5")
        
        # Manually set pattern position (Gazebo coordinates)
        self.pattern_pos = np.array([-2.8, 0.2, 0.0])  # Target pattern position
        
        # Initialize drone position from /odom
        self.drone_pos = np.array([0.0, 0.0, 0.0])
        
        # ROS 2 publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Control loop timer
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        self.prev_distance = np.inf  # For reward calculation

    def odom_callback(self, msg):
        """Update drone position from /odom topic"""
        self.drone_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.get_logger().info(f"Updated drone position: {self.drone_pos}")

    def calculate_reward(self, prev_distance, new_distance):
        """Calculate reward based on distance to pattern"""
        reward = 1.0 if new_distance < prev_distance else 0.0
        reward += -0.05 * new_distance
        if any(self.drone_pos[:2] <= 0.1) or any(self.drone_pos[:2] >= 4.9):
            reward -= 0.5
        return reward

    def control_loop(self):
        """Generate action, publish to /cmd_vel, and log reward"""
        # Construct state: [drone_pos / 100.0, pattern_pos / 100.0] (normalized as in ddpg.py)
        state = np.concatenate([self.drone_pos / 100.0, self.pattern_pos / 100.0])
        state = np.array(state).reshape(1, -1)
        
        # Predict action using trained actor
        action = self.actor.predict(state, verbose=0)[0]
        action = np.clip(action, -0.5, 0.5)  # Match ddpg.py clipping
        action_scaled = action * 1.0  # Scale action to velocity
        
        # Calculate reward based on current and previous distance
        new_distance = np.linalg.norm(self.drone_pos[:2] - self.pattern_pos[:2])
        reward = self.calculate_reward(self.prev_distance, new_distance)
        self.prev_distance = new_distance
        
        # Create and publish Twist message
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = float(action_scaled[0])
        cmd_vel_msg.linear.y = float(action_scaled[1])
        cmd_vel_msg.linear.z = 0.0
        cmd_vel_msg.angular.x = 0.0
        cmd_vel_msg.angular.y = 0.0
        cmd_vel_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel_msg)
        
        # Log action and reward
        self.get_logger().info(f"Action: [{action_scaled[0]:.2f}, {action_scaled[1]:.2f}], Reward: {reward:.2f}, Distance to pattern: {new_distance:.2f}")
        
        # Check if target reached
        if new_distance < 0.1:
            self.get_logger().info(f"Reached pattern at {self.pattern_pos} from {self.drone_pos}")
            self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = TestDDPGNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
