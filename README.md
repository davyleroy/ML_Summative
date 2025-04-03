# Rural Health Access Optimization System

## Overview

The Rural Health Access Optimization System is a reinforcement learning project that optimizes healthcare resource allocation in rural settings. The system dynamically matches patients with appropriate healthcare resources (facilities, community health workers, mobile clinics) based on urgency, availability, and geographical constraints specific to rural African settings.

This project implements and compares two reinforcement learning approaches:

- Deep Q-Network (DQN)
- Proximal Policy Optimization (PPO)

## Environment

![Environment Visualization](environment_visualization.png)

The environment is a grid-based simulation where:

- **Red Circles**: Patients (size/color intensity indicates urgency level)
- **Blue Squares**: Healthcare Facilities (color intensity shows capacity)
- **Green Triangles**: Community Health Workers (green when available, gray when unavailable)
- **Orange Diamonds**: Mobile Clinics (orange when available, gray when unavailable)

### Action Space

The agent can perform 8 discrete actions for each patient:

**Rural Healthcare Connection:**

1. Connect Patient to Nearest Facility
2. Link with Community Health Worker
3. Schedule Mobile Clinic Visit
4. Initiate Emergency Response

**Digital Health Management:** 5. Generate Digital Health Records 6. Process Insurance Claims 7. Coordinate Pharmacy Services 8. Update Stakeholder Information

### State Space

The state space includes:

- Patient positions and urgency levels
- Healthcare facility positions and capacities
- Health worker positions and availability
- Mobile clinic positions and availability
- Digital health management statuses

### Reward Structure

- **Positive Rewards**: +5 to +15 for successful patient connections, record creation, and stakeholder updates
- **Negative Rewards**: -2 to -10 for failed connections, poor resource allocation, and redundant actions
- **Terminal Bonuses**: Additional rewards for serving all patients, completing all records, and updating all stakeholders

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Ml_summative.git
cd Mm_Summative

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train DQN agent
python main.py --mode train --algorithm dqn --num_episodes 1000

# Train PPO agent
python main.py --mode train --algorithm ppo --num_episodes 1000
```

### Evaluation

```bash
# Evaluate DQN agent
python main.py --mode evaluate --algorithm dqn --model_path models/dqn/your_model_folder/best_model.pth --render

# Evaluate PPO agent
python main.py --mode evaluate --algorithm ppo --model_path models/ppo/your_model_folder/best_model.pth --render
```

### Visualization

```bash
# Just visualize the environment
python main.py --mode visualize
```

## Project Structure

```
ML_Summative/
├── environment/
│   ├── __init__.py
│   ├── custom_env.py            # Rural Health Environment implementation
│   ├── rendering.py             # Visualization components
│
├── training/
│   ├── __init__.py
│   ├── dqn_training.py          # DQN implementation
│   ├── pg_training.py           # PPO implementation
│   ├── utils.py                 # Shared utility functions
│
├── models/
│   ├── dqn/                     # Saved DQN models
│   └── pg/                      # Saved PPO models
│
├── results/
│   ├── videos/                  # Recorded agent videos
│   ├── plots/                   # Performance plots
│   └── logs/                    # Training logs
│
├── main.py                      # Entry point for running experiments
├── evaluate.py                  # Script for evaluating trained models
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Results

### DQN Performance

DQN showed stable learning with consistent performance around episode 450, successfully serving 5 patients with rewards between 8-21. The model demonstrated good generalization during evaluation.

### PPO Performance

PPO initially struggled but showed dramatic improvement after episode 770, eventually achieving much higher rewards (up to 1968.00) compared to DQN. By episodes 920-1000, PPO consistently achieved high rewards (average ~956.58) and served 3-5 patients per episode.

However, PPO showed poor generalization during evaluation, with significant performance drops compared to training.

## Dependencies

- Python 3.8+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- PyGame (for visualization)

## License

[MIT License](LICENSE)

## Acknowledgments

This project was developed as part of the ML Techniques II course at African Leadership University.
