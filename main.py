import os
import argparse
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import time
import json

from environment.custom_env import RuralHealthEnv
from training.dqn_training import train_dqn
from training.pg_training import train_ppo
from environment.rendering import render_rural_health_env

def parse_args():
    parser = argparse.ArgumentParser(description='Rural Health Access Optimization System')
    
    # General arguments
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'visualize'],
                        help='Mode to run the script in: train, evaluate, or visualize')
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'ppo'],
                        help='RL algorithm to use: dqn or ppo')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a saved model for evaluation or visualization')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    parser.add_argument('--record', action='store_true',
                        help='Record video during evaluation')
    
    # Environment parameters
    parser.add_argument('--grid_size', type=int, default=10,
                        help='Size of the grid environment')
    parser.add_argument('--num_patients', type=int, default=5,
                        help='Number of patients in the environment')
    parser.add_argument('--num_facilities', type=int, default=3,
                        help='Number of healthcare facilities in the environment')
    parser.add_argument('--num_health_workers', type=int, default=5,
                        help='Number of community health workers in the environment')
    parser.add_argument('--num_mobile_clinics', type=int, default=2,
                        help='Number of mobile clinics in the environment')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=500,
                        help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum number of steps per episode')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for future rewards')
    
    # DQN specific parameters
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help='Starting value of epsilon for exploration')
    parser.add_argument('--epsilon_end', type=float, default=0.01,
                        help='Minimum value of epsilon for exploration')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                        help='Decay rate of epsilon per episode')
    parser.add_argument('--buffer_size', type=int, default=10000,
                        help='Size of the replay buffer')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--target_update', type=int, default=10,
                        help='Number of steps between target network updates')
    
    # PPO specific parameters
    parser.add_argument('--lr_actor', type=float, default=0.0003,
                        help='Learning rate for the actor network')
    parser.add_argument('--lr_critic', type=float, default=0.001,
                        help='Learning rate for the critic network')
    parser.add_argument('--clip_ratio', type=float, default=0.2,
                        help='PPO clip ratio')
    parser.add_argument('--ppo_epochs', type=int, default=10,
                        help='Number of epochs to optimize on the same data')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='results/logs',
                        help='Directory to save logs')
    parser.add_argument('--video_dir', type=str, default='results/videos',
                        help='Directory to save videos')
    
    return parser.parse_args()

def create_environment(args, render_mode=None):
    """Create the Rural Health environment with specified parameters."""
    env = RuralHealthEnv(
        grid_size=args.grid_size,
        num_patients=args.num_patients,
        num_facilities=args.num_facilities,
        num_health_workers=args.num_health_workers,
        num_mobile_clinics=args.num_mobile_clinics,
        render_mode=render_mode
    )
    return env

def flatten_observation(observation):
    """Flatten the observation dictionary into a 1D array."""
    flattened = []
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            flattened.extend(value.flatten())
    return np.array(flattened)

def train(args):
    """Train an agent using the specified algorithm."""
    # Create environment
    env = create_environment(args)
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    
    # Create timestamp for unique directory names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up directories
    model_dir = os.path.join(args.save_dir, args.algorithm, timestamp)
    log_dir = os.path.join(args.log_dir, args.algorithm, timestamp)
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    print(f"Training {args.algorithm.upper()} for {args.num_episodes} episodes...")
    
    # Train the agent
    if args.algorithm == 'dqn':
        agent = train_dqn(
            env=env,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            target_update=args.target_update,
            save_dir=model_dir,
            log_dir=log_dir
        )
    elif args.algorithm == 'ppo':
        agent = train_ppo(
            env=env,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            clip_ratio=args.clip_ratio,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            save_dir=model_dir,
            log_dir=log_dir
        )
    
    print(f"Training complete. Models saved to {model_dir}")
    return agent

def evaluate(args):
    """Evaluate a trained agent."""
    # Determine render mode
    render_mode = "human" if args.render else None
    
    # Create environment
    env = create_environment(args, render_mode=render_mode)
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    
    # Check if model path is provided
    if args.model_path is None:
        raise ValueError("Model path must be provided for evaluation mode")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    if args.algorithm == 'dqn':
        from training.dqn_training import DQN
        
        # Get state and action sizes
        observation, _ = env.reset()
        state_size = len(flatten_observation(observation))
        action_size = env.action_space["action_type"].n * env.action_space["patient_id"].n
        
        # Create model and load weights
        model = DQN(state_size, action_size).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        
    elif args.algorithm == 'ppo':
        from training.pg_training import ActorCritic
        
        # Get state and action sizes
        observation, _ = env.reset()
        state_size = len(flatten_observation(observation))
        action_size = env.action_space["action_type"].n * env.action_space["patient_id"].n
        
        # Create model and load weights
        model = ActorCritic(state_size, action_size).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
    
    # Set up video recording if enabled
    if args.record:
        os.makedirs(args.video_dir, exist_ok=True)
        video_path = os.path.join(args.video_dir, f"{args.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        
        try:
            import imageio
            frames = []
        except ImportError:
            print("imageio not found. Video recording disabled.")
            args.record = False
    
    # Run evaluation episodes
    num_eval_episodes = 10
    rewards = []
    patients_served = []
    critical_cases_missed = []
    
    for episode in range(num_eval_episodes):
        observation, _ = env.reset()
        state = flatten_observation(observation)
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < args.max_steps:
            # Select action based on the algorithm
            if args.algorithm == 'dqn':
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = model(state_tensor)
                    action_idx = q_values.max(1)[1].item()
            else:  # PPO
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action_probs, _ = model(state_tensor)
                    action_idx = torch.argmax(action_probs).item()
            
            # Convert flat action index to dict action
            patient_id = action_idx % env.action_space["patient_id"].n
            action_type = action_idx // env.action_space["patient_id"].n
            action = {"patient_id": patient_id, "action_type": action_type}
            
            # Take action
            next_observation, reward, done, _, info = env.step(action)
            next_state = flatten_observation(next_observation)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Render if enabled
            if args.render:
                env.render()
                time.sleep(0.1)  # Slow down rendering for visibility
            
            # Record frame if enabled
            if args.record:
                frame = render_rural_health_env(env)
                frames.append(frame)
            
            step += 1
        
        # Store episode results
        rewards.append(episode_reward)
        patients_served.append(info['patients_served'])
        critical_cases_missed.append(info['critical_cases_missed'])
        
        print(f"Episode {episode+1}/{num_eval_episodes} | "
              f"Reward: {episode_reward:.2f} | "
              f"Patients Served: {info['patients_served']}/{args.num_patients} | "
              f"Critical Cases Missed: {info['critical_cases_missed']}")
    
    # Save video if recording
    if args.record and len(frames) > 0:
        print(f"Saving video to {video_path}")
        imageio.mimsave(video_path, frames, fps=10)
    
    # Print evaluation summary
    print("\nEvaluation Summary:")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Patients Served: {np.mean(patients_served):.2f} ± {np.std(patients_served):.2f}")
    print(f"Average Critical Cases Missed: {np.mean(critical_cases_missed):.2f} ± {np.std(critical_cases_missed):.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.bar(range(num_eval_episodes), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode')
    
    plt.subplot(1, 3, 2)
    plt.bar(range(num_eval_episodes), patients_served)
    plt.xlabel('Episode')
    plt.ylabel('Patients Served')
    plt.title('Patients Served per Episode')
    
    plt.subplot(1, 3, 3)
    plt.bar(range(num_eval_episodes), critical_cases_missed)
    plt.xlabel('Episode')
    plt.ylabel('Critical Cases Missed')
    plt.title('Critical Cases Missed per Episode')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results/plots', exist_ok=True)
    plot_path = os.path.join('results/plots', f"{args.algorithm}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path)
    print(f"Evaluation plot saved to {plot_path}")
    
    if args.render:
        plt.show()

def visualize(args):
    """Visualize the environment without training."""
    # Create environment with rendering
    env = create_environment(args, render_mode="human")
    
    # Reset environment
    env.reset(seed=args.seed)
    
    print("Visualizing environment. Press Ctrl+C to exit.")
    
    try:
        # Just render the environment without any agent actions
        while True:
            env.render()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
    finally:
        env.close()

def main():
    """Main entry point of the program."""
    args = parse_args()
    
    # Create necessary directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'visualize':
        visualize(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
    # Ensure the script is run from the correct directory