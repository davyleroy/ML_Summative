import gymnasium as gym
import numpy as np
import torch
import argparse
import os
import sys
import time
from collections import deque
import matplotlib.pyplot as plt
import imageio
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env import RuralHealthEnv
from environment.rendering import render_rural_health_env
from training.dqn_training import DQN, flatten_observation
from training.pg_training import ActorCritic

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained RL models')
    parser.add_argument('--model_type', type=str, default='dqn', choices=['dqn', 'pg'],
                        help='Type of model to evaluate (dqn or pg)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment')
    parser.add_argument('--save_video', action='store_true',
                        help='Save a video of the agent performance')
    parser.add_argument('--video_path', type=str, default='results/videos/agent.mp4',
                        help='Path to save the video')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()

def evaluate_dqn(env, model_path, num_episodes=10, render=False, save_video=False, video_path=None, seed=42):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get state and action sizes
    observation, _ = env.reset(seed=seed)
    state_size = len(flatten_observation(observation))
    action_size = env.action_space["action_type"].n * env.action_space["patient_id"].n
    
    # Initialize model
    model = DQN(state_size, action_size).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Evaluation metrics
    episode_rewards = []
    patients_served_list = []
    critical_cases_missed_list = []
    
    # For video recording
    if save_video and video_path:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        frames = []
    
    for episode in range(num_episodes):
        observation, _ = env.reset(seed=seed + episode)
        state = flatten_observation(observation)
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            # Select action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action_idx = q_values.max(1)[1].item()
            
            # Convert flat action index to dict action
            patient_id = action_idx % env.action_space["patient_id"].n
            action_type = action_idx // env.action_space["patient_id"].n
            action = {"patient_id": patient_id, "action_type": action_type}
            
            # Take action
            next_observation, reward, done, _, info = env.step(action)
            next_state = flatten_observation(next_observation)
            
            # Render if requested
            if render or save_video:
                frame = render_rural_health_env(env)
                if save_video:
                    frames.append(frame)
                if render:
                    plt.imshow(frame)
                    plt.pause(0.1)
                    plt.clf()
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            step += 1
            
            if done:
                break
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        patients_served_list.append(info['patients_served'])
        critical_cases_missed_list.append(info['critical_cases_missed'])
        
        print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | "
              f"Patients Served: {info['patients_served']} | "
              f"Critical Cases Missed: {info['critical_cases_missed']}")
    
    # Save video if requested
    if save_video and video_path and frames:
        print(f"Saving video to {video_path}...")
        imageio.mimsave(video_path, [Image.fromarray(frame) for frame in frames], fps=10)
    
    # Print evaluation summary
    print("\nEvaluation Summary:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Patients Served: {np.mean(patients_served_list):.2f} ± {np.std(patients_served_list):.2f}")
    print(f"Average Critical Cases Missed: {np.mean(critical_cases_missed_list):.2f} ± {np.std(critical_cases_missed_list):.2f}")
    
    return {
        'episode_rewards': episode_rewards,
        'patients_served': patients_served_list,
        'critical_cases_missed': critical_cases_missed_list
    }

def evaluate_pg(env, model_path, num_episodes=10, render=False, save_video=False, video_path=None, seed=42):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get state and action sizes
    observation, _ = env.reset(seed=seed)
    state_size = len(flatten_observation(observation))
    action_size = env.action_space["action_type"].n * env.action_space["patient_id"].n
    
    # Initialize model
    model = ActorCritic(state_size, action_size).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Evaluation metrics
    episode_rewards = []
    patients_served_list = []
    critical_cases_missed_list = []
    
    # For video recording
    if save_video and video_path:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        frames = []
    
    for episode in range(num_episodes):
        observation, _ = env.reset(seed=seed + episode)
        state = flatten_observation(observation)
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            # Select action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action_probs, _ = model(state_tensor)
                action_idx = torch.argmax(action_probs, dim=1).item()
            
            # Convert flat action index to dict action
            patient_id = action_idx % env.action_space["patient_id"].n
            action_type = action_idx // env.action_space["patient_id"].n
            action = {"patient_id": patient_id, "action_type": action_type}
            
            # Take action
            next_observation, reward, done, _, info = env.step(action)
            next_state = flatten_observation(next_observation)
            
            # Render if requested
            if render or save_video:
                frame = render_rural_health_env(env)
                if save_video:
                    frames.append(frame)
                if render:
                    plt.imshow(frame)
                    plt.pause(0.1)
                    plt.clf()
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            step += 1
            
            if done:
                break
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        patients_served_list.append(info['patients_served'])
        critical_cases_missed_list.append(info['critical_cases_missed'])
        
        print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | "
              f"Patients Served: {info['patients_served']} | "
              f"Critical Cases Missed: {info['critical_cases_missed']}")
    
    # Save video if requested
    if save_video and video_path and frames:
        print(f"Saving video to {video_path}...")
        imageio.mimsave(video_path, [Image.fromarray(frame) for frame in frames], fps=10)
    
    # Print evaluation summary
    print("\nEvaluation Summary:")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Patients Served: {np.mean(patients_served_list):.2f} ± {np.std(patients_served_list):.2f}")
    print(f"Average Critical Cases Missed: {np.mean(critical_cases_missed_list):.2f} ± {np.std(critical_cases_missed_list):.2f}")
    
    return {
        'episode_rewards': episode_rewards,
        'patients_served': patients_served_list,
        'critical_cases_missed': critical_cases_missed_list
    }

def plot_comparison(dqn_results, pg_results, save_path='results/plots/comparison.png'):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Set up figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot rewards
    axs[0].boxplot([dqn_results['episode_rewards'], pg_results['episode_rewards']], 
                  labels=['DQN', 'PPO'])
    axs[0].set_title('Episode Rewards')
    axs[0].set_ylabel('Reward')
    
    # Plot patients served
    axs[1].boxplot([dqn_results['patients_served'], pg_results['patients_served']], 
                  labels=['DQN', 'PPO'])
    axs[1].set_title('Patients Served')
    axs[1].set_ylabel('Count')
    
    # Plot critical cases missed
    axs[2].boxplot([dqn_results['critical_cases_missed'], pg_results['critical_cases_missed']], 
                  labels=['DQN', 'PPO'])
    axs[2].set_title('Critical Cases Missed')
    axs[2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Comparison plot saved to {save_path}")

if __name__ == "__main__":
    args = parse_args()
    
    # Create environment
    env = RuralHealthEnv(grid_size=10, num_patients=5, num_facilities=3, 
                         num_health_workers=5, num_mobile_clinics=2,
                         render_mode='rgb_array' if args.render or args.save_video else None)
    
    # Evaluate model
    if args.model_type == 'dqn':
        results = evaluate_dqn(
            env=env,
            model_path=args.model_path,
            num_episodes=args.num_episodes,
            render=args.render,
            save_video=args.save_video,
            video_path=args.video_path,
            seed=args.seed
        )
    elif args.model_type == 'pg':
        results = evaluate_pg(
            env=env,
            model_path=args.model_path,
            num_episodes=args.num_episodes,
            render=args.render,
            save_video=args.save_video,
            video_path=args.video_path,
            seed=args.seed
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")