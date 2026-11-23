

https://github.com/user-attachments/assets/4fe6611c-ad72-4e68-a4c4-58168465d1c8

# DQN Agent for Atari Assault

A Deep Q-Network (DQN) agent that learns to play the Atari 2600 game **Assault** (`ALE/Assault-v5`) straight from screen pixels. It is built with [Stable-Baselines3] and [Gymnasium]. After training, the notebook records videos of the agent playing.

## How it works

DQN learns a function **Q(state, action)** that estimates the future reward of taking each action from the current screen. The agent picks the action with the highest Q-value. Training relies on three standard DQN ideas:

- **A CNN policy**: a convolutional network reads stacked game frames and outputs one Q-value per action.
- **Experience replay**: past transitions go into a buffer, and the network trains on random mini-batches drawn from it. This breaks the correlation between consecutive frames.
- **A target network**: a separate, slowly updated copy of the network supplies stable training targets.
- **E-greedy exploration**: the agent starts by acting fully at random (E = 1.0) and gradually shifts to acting on what it has learned (E = 0.01).

The frames are preprocessed with SB3's AtariWrapper: grayscale, resizing to 84×84, frame skipping, no-op resets, a life lost counted as the end of an episode, and reward clipping.

## Project structure

```
DQN_game/
├── assult_game.ipynb     # Training + video recording notebook
├── requirements.txt      # Python dependencies
├── README.md
├── assault_dqn_model/    # saved models
└── demo_video/           # recorded gameplay .mp4 files
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

The Atari game files come bundled with ale-py (version 0.9 and later), so no separate download is needed.

## Usage

Open `assult_game.ipynb` and run the cells in order.

### 1. Imports
This cell loads Gymnasium, Stable-Baselines3, PyTorch and ALE.

### 2. Training
- Builds the Assault environment with `Monitor` and `AtariWrapper`.
- Trains the DQN agent and prints the mean reward of the last 10 episodes every 10,000 steps.
- Saves the best model seen during training as `assault_dqn_model/best_assault_model.zip`.
- Saves the final model as `assault_dqn_model/final_assault_model.zip`.
- You can stop training safely with **Interrupt Kernel**. The model trained up to that point is still saved.

### 3. Video recording
- Loads `best_assault_model`, or `final_assault_model` if no best model exists.
- Runs 3 episodes with the agent always choosing its best action (`deterministic=True`), capped at 3,000 steps each.
- Saves the videos as `demo_video/dqn_assault_gameplay-episode-*.mp4` and prints the score and step count of each episode.

## Hyperparameters

| Parameter | GPU | CPU |
|---|---|---|
| Total timesteps | 500,000 | 100,000 |
| Replay buffer size | 100,000 | 50,000 |
| Learning starts | 50,000 | 50,000 |
| Batch size | 32 | 32 |
| Learning rate | 1e-4 | 1e-4 |
| Discount (y) | 0.99 | 0.99 |
| Train frequency | every 4 steps | every 4 steps |
| Target update interval | 10,000 | 10,000 |
| Exploration | 1.0 → 0.01 over first 10% | same |

The notebook detects CUDA automatically and picks the matching settings.

## Expected results

- **Steps 0–50k:** the agent mostly acts at random while the replay buffer fills, so rewards stay low.
- **After 50k:** learning starts, and the mean reward should climb gradually.
- On CPU, the run is only 100k steps and half of them are random warm-up, so expect a modest agent. For noticeably better play, train for 1M+ steps on a GPU.
