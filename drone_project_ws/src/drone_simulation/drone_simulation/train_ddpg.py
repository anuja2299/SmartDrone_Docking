import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import torch
import torch.nn as nn
import torch.optim as optim

# Define Actor and Critic Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class DDPGAgent(Node):
    def __init__(self):
        super().__init__('ddpg_agent')
        self.get_logger().info("DDPG Agent initializing...")
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.get_logger().info("Subscribed to /odom")
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("Publisher for /cmd_vel created")
        self.timer = self.create_timer(0.1, self.train_step)
        self.get_logger().info("Timer set for train_step")
        self.drone_pos = None
        self.target_pos = np.array([-2.8, 0.2])  # Hardcoded target position
        self.memory = []
        self.batch_size = 32
        self.gamma = 0.99
        self.actor = Actor(state_dim=2, action_dim=2)
        self.critic = Critic(state_dim=2, action_dim=2)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def odom_callback(self, msg):
        self.drone_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.get_logger().info(f"Received odom: {self.drone_pos}, frame_id: {msg.header.frame_id}")

    def get_state(self):
        if self.drone_pos is None:
            return None
        state = self.target_pos - np.array(self.drone_pos)  # Relative position to target
        self.get_logger().info(f"Computed state: {state}")
        return state

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]
        return np.clip(action * 2.0, -2.0, 2.0)  # Scale actions to [-2, 2]

    def get_reward(self, prev_state, new_state):
        prev_distance = np.linalg.norm(prev_state)
        new_distance = np.linalg.norm(new_state)
        reward = 1.0 if new_distance < prev_distance else 0.0  # Positive reward if distance decreases
        self.get_logger().info(f"Reward: {reward}, prev_distance: {prev_distance}, new_distance: {new_distance}")
        return reward

    def train_step(self):
        state = self.get_state()
        self.get_logger().info(f"State check: {state}")
        if state is None:
            self.get_logger().info("State is None, skipping step")
            return

        action = self.get_action(state)
        # Add noise for exploration
        noise = np.random.normal(0, 0.2, 2)  # Increased std to 0.2
        action += noise
        action = np.clip(action, -2.0, 2.0)  # Ensure within bounds
        new_state = self.get_state()  # Update state after action (simplified)
        reward = self.get_reward(state, new_state)

        self.get_logger().info(f"Drone pos: {self.drone_pos}, Target pos: {self.target_pos.tolist()}, Relative: {state}, Reward: {reward}, Action: {action}")

        # Store transition
        self.memory.append((state, action, reward, new_state, False))
        if len(self.memory) > self.batch_size:
            self.update_model()

        # Publish action with explicit float conversion
        msg = Twist()
        msg.linear.x = float(action[0])  # Convert to Python float
        msg.linear.y = float(action[1])  # Convert to Python float
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)

    def update_model(self):
        if len(self.memory) < self.batch_size:
            return
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Critic loss
        target_q = rewards + self.gamma * self.critic(next_states, self.actor(next_states)) * (1 - dones)
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

def main(args=None):
    rclpy.init(args=args)
    agent = DDPGAgent()
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
