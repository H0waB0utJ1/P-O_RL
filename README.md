# P-O_RL

P-O_RL is a reinforcement learning project for training locomotion policies for the Unitree G1 humanoid robot in MuJoCo. The repository focuses on a practical PPO training loop with parallel rollout collection, configurable reward shaping, and a lightweight inference entrypoint for evaluating trained policies.

## Overview

This project implements a humanoid locomotion stack built around:

- `MuJoCo` for rigid-body simulation
- `PPO` for on-policy policy optimization
- `Ray` for parallel trajectory sampling
- `PyTorch` for actor/critic modeling and optimization

The current setup targets a 29-DoF Unitree G1 model and is primarily oriented toward forward walking under command-conditioned reward shaping.

## Highlights

- Parallel rollout collection with Ray workers
- Feed-forward and recurrent (LSTM) PPO policies
- PD-controlled action interface on top of MuJoCo simulation
- Centralized reward and environment configuration
- TensorBoard logging and periodic checkpoint export
- Separate training and inference entrypoints

## Repository Layout

```text
P-O_RL/
├── envs/
│   ├── env_config.py      # Environment, command, normalization, control, reward scales
│   ├── mujoco_env.py      # MuJoCo environment and simulation loop
│   ├── reward.py          # Reward term implementation
│   └── robot_interface.py # Robot state, kinematics, contacts, and torque limits
├── models/unitree_g1/     # MuJoCo XML assets and meshes for Unitree G1
├── rl/
│   ├── policies/          # Actor and critic networks
│   └── ppo/               # PPO trainer, rollout buffer, and config
├── scripts/
│   └── export_tb.py       # Utility script for exporting TensorBoard scalars
├── train.py               # Training entrypoint
├── run.py                 # Evaluation / playback entrypoint
└── README.md
```

## Environment Requirements

- Python 3.10 or newer
- MuJoCo Python bindings
- PyTorch
- Ray
- NumPy
- SciPy
- TensorBoard

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch ray numpy scipy mujoco tensorboard
```

If you are running on a headless machine, MuJoCo may require additional EGL or OSMesa configuration depending on your system.

## Quick Start

### 1. Train a policy

```bash
python train.py \
  --env models/unitree_g1/unitree_g1_es.xml \
  --num-procs 16 \
  --n-itr 500 \
  --max-traj-len 3000
```

Training logs and checkpoints are written under `runs/train/exp*/`.

### 2. Resume training

```bash
python train.py \
  --env models/unitree_g1/unitree_g1_es.xml \
  --continued runs/train/exp1/actor.pt \
  --num-procs 16 \
  --n-itr 300
```

### 3. Run evaluation

Using an experiment directory:

```bash
python run.py \
  --env models/unitree_g1/unitree_g1_es.xml \
  --path runs/train/exp1
```

Using a direct actor checkpoint:

```bash
python run.py \
  --env models/unitree_g1/unitree_g1_es.xml \
  --path runs/train/exp1/actor.pt
```

`run.py` resolves the matching critic checkpoint from the same directory automatically.

## Training Outputs

Each experiment directory typically contains:

- `actor.pt`, `critic.pt`: latest or best exported checkpoints
- `actor_<itr>.pt`, `critic_<itr>.pt`: periodic snapshot checkpoints
- `experiment.pkl`: serialized training arguments
- `events.out.tfevents.*`: TensorBoard event files

Launch TensorBoard with:

```bash
tensorboard --logdir runs/train
```

## Important Runtime Configuration

Most behavior is controlled from the following files:

- `train.py`: PPO hyperparameters and runtime flags
- `envs/env_config.py`: reward scales, command ranges, control gains, observation normalization
- `envs/reward.py`: reward term definitions
- `envs/mujoco_env.py`: simulation loop, reset logic, action processing, termination logic

Useful parameters to tune first:

- `--num-procs`
- `--n-itr`
- `--max-traj-len`
- `--epochs`
- `--minibatch-size`
- `tracking_lin_vel`
- `tracking_ang_vel`
- `orientation`
- `base_height`
- `action_rate`
- `torques`
