import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
from collections import deque, namedtuple
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import RuralHealthEnv

# Define DQN Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# Define Replay Buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, device, 
                 learning_rate=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=10):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon_start  # Exploration rate
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training step counter
        self.steps_done = 0
    
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.action_size)
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Create tensors
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                     device=self.device, dtype=torch.bool)
        non_final_next_states = torch.FloatTensor([s for s in batch.next_state if s is not None]).to(self.device)
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values * (1 - done_batch))
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.steps_done += 1
        
        return loss.item()

# Flatten observation for DQN input
def flatten_observation(observation):
    flattened = []
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            flattened.extend(value.flatten())
    return np.array(flattened)

# Training function
def train_dqn(env, num_episodes=1000, max_steps=100, 
              learning_rate=0.001, gamma=0.99, 
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
              buffer_size=10000, batch_size=64, target_update=10,
              save_dir='models/dqn', log_dir='logs/dqn'):
    
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get state and action sizes
    observation, _ = env.reset()
    state_size = len(flatten_observation(observation))
    action_size = env.action_space["action_type"].n * env.action_space["patient_id"].n
    
    # Initialize agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        device=device,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update=target_update
    )
    
    # Set up tensorboard
    writer = SummaryWriter(log_dir)
    
    # Training loop
    best_reward = -float('inf')
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        state = flatten_observation(observation)
        
        episode_reward = 0
        episode_loss = 0
        
        for step in range(max_steps):
            # Select action
            action_idx = agent.select_action(state)
            
            # Convert flat action index to dict action
            patient_id = action_idx % env.action_space["patient_id"].n
            action_type = action_idx // env.action_space["patient_id"].n
            action = {"patient_id": patient_id, "action_type": action_type}
            
            # Take action
            next_observation, reward, done, _, info = env.step(action)
            next_state = flatten_observation(next_observation)
            
            # Store transition
            agent.memory.push(state, action_idx, next_state, reward, done)
            
            # Learn
            loss = agent.learn()
            if loss is not None:
                episode_loss += loss
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update exploration rate
        agent.update_epsilon()
        
        # Log metrics
        writer.add_scalar('Reward/Episode', episode_reward, episode)
        writer.add_scalar('Loss/Episode', episode_loss, episode)
        writer.add_scalar('Epsilon/Episode', agent.epsilon, episode)
        writer.add_scalar('PatientsServed/Episode', info['patients_served'], episode)
        writer.add_scalar('CriticalCasesMissed/Episode', info['critical_cases_missed'], episode)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.2f} | Patients Served: {info['patients_served']}")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.policy_net.state_dict(), f"{save_dir}/best_model.pth")
        
        # Save checkpoint
        if (episode + 1) % 100 == 0:
            torch.save({
                'episode': episode,
                'policy_net_state_dict': agent.policy_net.state_dict(),
                'target_net_state_dict': agent.target_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'best_reward': best_reward
            }, f"{save_dir}/checkpoint_{episode+1}.pth")
    
    # Save final model
    torch.save(agent.policy_net.state_dict(), f"{save_dir}/final_model.pth")
    
    writer.close()
    
    return agent

if __name__ == "__main__":
    # Create environment
    env = RuralHealthEnv(grid_size=10, num_patients=5, num_facilities=3, 
                         num_health_workers=5, num_mobile_clinics=2)
    
    # Train DQN agent
    agent = train_dqn(
        env=env,
        num_episodes=1000,
        max_steps=100,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update=10
    )