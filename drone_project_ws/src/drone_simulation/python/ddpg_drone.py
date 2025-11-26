import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import json
import time
import sys

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.5):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self, episode):
        if episode > 100:  # Extended noise decay to 100 episodes
            return np.zeros(self.action_dimension)
        current_sigma = self.sigma * (1 - min(episode / 100, 1))
        x = self.state
        dx = self.theta * (self.mu - x) + current_sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer = []
        self.index = 0

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.index] = experience
        self.index = (self.index + 1) % self.buffer_size

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        states = np.zeros((batch_size, self.state_dim))
        actions = np.zeros((batch_size, self.action_dim))
        rewards = np.zeros(batch_size)
        next_states = np.zeros((batch_size, self.state_dim))
        dones = np.zeros(batch_size)
        for i, idx in enumerate(indices):
            s, a, r, ns, d = self.buffer[idx]
            states[i] = s
            actions[i] = a
            rewards[i] = r
            next_states[i] = ns
            dones[i] = d
        return states, actions, rewards, next_states, dones

class DDPG:
    def __init__(self, state_dim=6, action_dim=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.tau = 0.005  # Increased for faster updates
        self.batch_size = 32  # Reduced for more frequent updates
        self.actor = self.build_actor()
        self.actor_target = self.build_actor()
        self.critic = self.build_critic()
        self.critic_target = self.build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(0.0005)  # Adjusted learning rate
        self.critic_optimizer = tf.keras.optimizers.Adam(0.001)  # Adjusted learning rate
        self.noise = OUNoise(action_dim)
        self.buffer = ReplayBuffer(10000, state_dim, action_dim)
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())
        self.drone_pos = np.array([0.0, 0.0, 0.0])
        self.last_target_pos = np.array([2.8, 0.2, 0.0])  # Target position
        self.episode_rewards = []
        self.html_output = []

    def build_actor(self):
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.state_dim),
            layers.LayerNormalization(),
            layers.Dense(64, activation='relu'),
            layers.LayerNormalization(),
            layers.Dense(self.action_dim, activation='tanh')
        ])
        return model

    def build_critic(self):
        state_input = layers.Input(shape=(self.state_dim,))
        action_input = layers.Input(shape=(self.action_dim,))
        state_h = layers.Dense(128, activation='relu')(state_input)
        state_h = layers.LayerNormalization()(state_h)
        action_h = layers.Dense(128, activation='relu')(action_input)
        action_h = layers.LayerNormalization()(action_h)
        combined = layers.Concatenate()([state_h, action_h])
        h = layers.Dense(64, activation='relu')(combined)
        h = layers.LayerNormalization()(h)
        q = layers.Dense(1)(h)
        model = models.Model([state_input, action_input], q)
        return model

    def update_target(self, target_weights, weights):
        for tw, w in zip(target_weights, weights):
            tw.assign(tw * (1 - self.tau) + w * self.tau)

    def act(self, state):
        state = np.array(state).reshape(1, -1)
        action = self.actor.predict(state, verbose=0)[0]
        return np.clip(action, -0.5, 0.5)

    def train(self):
        if len(self.buffer.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        with tf.GradientTape() as tape:
            target_actions = self.actor_target(next_states)
            target_q = self.critic_target([next_states, target_actions])
            y = rewards + self.gamma * target_q * (1 - dones)
            q = self.critic([states, actions])
            critic_loss = tf.reduce_mean(tf.square(y - q))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic([states, new_actions]))
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.update_target(self.actor_target.weights, self.actor.weights)
        self.update_target(self.critic_target.weights, self.critic.weights)

    def get_pattern_position(self):
        for _ in range(3):
            try:
                with open("pattern_position.txt", "r") as f:
                    content = f.read().strip()
                    if not content:
                        self.custom_print("Warning: Pattern position file is empty. Using last known position.")
                        return self.last_target_pos
                    pos = json.loads(content)
                    self.last_target_pos = np.array([float(pos["x"]), float(pos["y"]), float(pos["z"])])
                    return self.last_target_pos
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                self.custom_print(f"Error reading pattern position: {e}. Retrying...")
                time.sleep(0.05)
        self.custom_print("Failed to read pattern position. Using last known position.")
        return self.last_target_pos

    def custom_print(self, *args, **kwargs):
        message = " ".join(str(arg) for arg in args)
        self.html_output.append(message)
        print(message, **kwargs)

    def run(self):
        self.custom_print("Starting DDPG algorithm...")
        episode = 0
        max_episodes = 500  # Increased to 500 episodes
        with open("ddpg_rewards.txt", "w") as f:
            f.write("episode,reward\n")
        with open("ddpg_positions.txt", "w") as p:
            p.write("episode,x,y,z\n")
        with open("ddpg_output.html", "w") as f:
            f.write("<html><head><style>body {font-family: Arial, sans-serif; background-color: #f0f0f0; color: #333;}</style></head><body><h2>DDPG Training Log</h2><pre>")
        while episode < max_episodes:
            episode += 1
            self.drone_pos = np.array([0.0, 0.0, 0.0])
            self.noise.reset()
            step = 0
            episode_reward = 0
            while True:
                step += 1
                target_pos = self.get_pattern_position()
                state = np.concatenate([self.drone_pos / 100.0, target_pos / 100.0])
                action = self.act(state)
                if step <= 5:
                    action = np.random.uniform(-0.5, 0.5, self.action_dim)
                action += self.noise.noise(episode)
                action = np.clip(action, -0.5, 0.5)
                action_scaled = action * 10.0
                prev_distance = np.linalg.norm(self.drone_pos[:2] - target_pos[:2])
                self.drone_pos[:2] += action_scaled
                self.drone_pos[2] = 0
                self.drone_pos[:2] = np.clip(self.drone_pos[:2], 0, 99)  # Adjusted bounds to 0-99
                new_distance = np.linalg.norm(self.drone_pos[:2] - target_pos[:2])
                # Enhanced reward function
                reward = 5.0 if new_distance < 1.0 else 2.0 if new_distance < 5.0 else -1.0 if new_distance > 20.0 else 0.0
                reward += -0.01 * new_distance / 70
                if any(self.drone_pos[:2] <= 2) or any(self.drone_pos[:2] >= 98):
                    reward -= 0.5
                done = new_distance < 1.0
                episode_reward += reward
                next_state = np.concatenate([self.drone_pos / 100.0, target_pos / 100.0])
                self.buffer.add(state, action, reward, next_state, done)
                self.train()
                self.custom_print(f"Episode: {episode}, Step: {step}, Drone Pos: ({self.drone_pos[0]:.2f}, {self.drone_pos[1]:.2f}, {self.drone_pos[2]:.2f}), Target: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}), Reward: {reward:.2f}, Distance: {new_distance:.2f}, Episode Reward: {episode_reward:.2f}, Action: ({action_scaled[0]:.2f}, {action_scaled[1]:.2f})")
                if done or step > 50:
                    self.custom_print(f"Episode {episode} completed at step {step}, Episode Reward: {episode_reward:.2f}, Final Position: {self.drone_pos}")
                    with open("ddpg_rewards.txt", "a") as f:
                        f.write(f"{episode},{episode_reward}\n")
                    with open("ddpg_positions.txt", "a") as p:
                        p.write(f"{episode},{self.drone_pos[0]},{self.drone_pos[1]},{self.drone_pos[2]}\n")
                    self.episode_rewards.append(episode_reward)
                    break
                time.sleep(0.01)  # Reduced sleep time
        self.custom_print(f"Training completed. Final Reward: {self.episode_rewards[-1]:.2f}")
        with open("ddpg_output.html", "a") as f:
            f.write("</pre></body></html>")

def main():
    ddpg = DDPG()
    ddpg.run()
    # Generate and save the plot after training
    import matplotlib.pyplot as plt
    episodes = range(1, len(ddpg.episode_rewards) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(episodes, ddpg.episode_rewards, 'b-', label='Average Reward')
    plt.axhline(y=0, color='k', linestyle='--', label='Zero Reward')
    plt.title('DDPG Training Reward Convergence (500 Episodes)')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.legend()
    plt.savefig('ddpg_reward_convergence.png')
    plt.show()

if __name__ == "__main__":
    main()
