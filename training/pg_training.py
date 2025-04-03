import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from collections import deque

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import RuralHealthEnv

# Define Actor-Critic Network for PPO
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value
    
    def get_action(self, state, device):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs, _ = self.forward(state)
        
        # Create a distribution and sample
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), action_probs[0][action.item()].item()

# PPO Agent
class PPOAgent:
    def __init__(self, state_size, action_size, device,
                 lr_actor=0.0003, lr_critic=0.001, gamma=0.99, 
                 clip_ratio=0.2, ppo_epochs=10, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Actor-Critic network
        self.ac_model = ActorCritic(state_size, action_size).to(device)
        
        # Separate optimizers for actor and critic
        self.optimizer = optim.Adam([
            {'params': self.ac_model.shared.parameters()},
            {'params': self.ac_model.actor.parameters(), 'lr': lr_actor},
            {'params': self.ac_model.critic.parameters(), 'lr': lr_critic}
        ])
        
        # Memory for storing trajectories
        self.states = []
        self.actions = []
        self.action_probs = []  # Log probs of actions taken
        self.rewards = []
        self.dones = []
        self.values = []
    
    def select_action(self, state):
        action, action_prob = self.ac_model.get_action(state, self.device)
        
        # Get value estimate
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.ac_model(state_tensor)
        
        # Store trajectory information
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.values.append(value.item())
        
        return action
    
    def store_transition(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)
    
    def learn(self):
        # Convert lists to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_action_probs = torch.FloatTensor(self.action_probs).to(self.device)
        
        # Compute returns and advantages
        returns = self._compute_returns()
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        values_tensor = torch.FloatTensor(self.values).to(self.device)
        advantages = returns_tensor - values_tensor
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        # Mini-batch training
        batch_size = min(self.batch_size, len(self.states))
        indices = np.arange(len(self.states))
        
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(self.states), batch_size):
                end_idx = min(start_idx + batch_size, len(self.states))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_probs = old_action_probs[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                action_probs, values = self.ac_model(batch_states)
                
                # Create distributions
                dist = Categorical(action_probs)
                
                # Get log probs of actions
                new_action_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute ratio
                ratio = torch.exp(new_action_probs - torch.log(batch_old_probs + 1e-10))
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                
                # Actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                critic_loss = F.mse_loss(values.squeeze(-1), batch_returns)
                
                # Total loss
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(), 0.5)
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
        
        # Clear memory
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
        # Return average losses
        avg_actor_loss = total_actor_loss / (self.ppo_epochs * (len(indices) // batch_size + 1))
        avg_critic_loss = total_critic_loss / (self.ppo_epochs * (len(indices) // batch_size + 1))
        avg_entropy = total_entropy / (self.ppo_epochs * (len(indices) // batch_size + 1))
        
        return avg_actor_loss, avg_critic_loss, avg_entropy
    
    def _compute_returns(self):
        returns = []
        
        # Compute discounted returns
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        
        return returns

# Flatten observation for PPO input
def flatten_observation(observation):
    flattened = []
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            flattened.extend(value.flatten())
    return np.array(flattened)

# Training function
def train_ppo(env, num_episodes=1000, max_steps=100, 
              lr_actor=0.0003, lr_critic=0.001, gamma=0.99,
              clip_ratio=0.2, ppo_epochs=10, batch_size=64,
              save_dir='models/pg', log_dir='logs/pg'):
    
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
    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
        device=device,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        clip_ratio=clip_ratio,
        ppo_epochs=ppo_epochs,
        batch_size=batch_size
    )
    
    # Set up tensorboard
    writer = SummaryWriter(log_dir)
    
    # Training loop
    best_reward = -float('inf')
    recent_rewards = deque(maxlen=100)
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        state = flatten_observation(observation)
        
        episode_reward = 0
        
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
            agent.store_transition(reward, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Store episode reward
        recent_rewards.append(episode_reward)
        
        # Learn from episode
        if (episode + 1) % 5 == 0:  # Update every 5 episodes
            actor_loss, critic_loss, entropy = agent.learn()
            
            # Log metrics
            writer.add_scalar('Loss/Actor', actor_loss, episode)
            writer.add_scalar('Loss/Critic', critic_loss, episode)
            writer.add_scalar('Entropy', entropy, episode)
        
        # Log episode metrics
        writer.add_scalar('Reward/Episode', episode_reward, episode)
        writer.add_scalar('Reward/Average100', np.mean(recent_rewards), episode)
        writer.add_scalar('PatientsServed/Episode', info['patients_served'], episode)
        writer.add_scalar('CriticalCasesMissed/Episode', info['critical_cases_missed'], episode)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | "
                  f"Avg Reward: {avg_reward:.2f} | Patients Served: {info['patients_served']}")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.ac_model.state_dict(), f"{save_dir}/best_model.pth")
        
        # Save checkpoint
        if (episode + 1) % 100 == 0:
            torch.save({
                'episode': episode,
                'model_state_dict': agent.ac_model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'best_reward': best_reward
            }, f"{save_dir}/checkpoint_{episode+1}.pth")
    
    # Save final model
    torch.save(agent.ac_model.state_dict(), f"{save_dir}/final_model.pth")
    
    writer.close()
    
    return agent

if __name__ == "__main__":
    # Create environment
    env = RuralHealthEnv(grid_size=10, num_patients=5, num_facilities=3, 
                         num_health_workers=5, num_mobile_clinics=2)
    
    # Train PPO agent
    agent = train_ppo(
        env=env,
        num_episodes=500,
        max_steps=100,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        clip_ratio=0.2,
        ppo_epochs=10,
        batch_size=64
    )