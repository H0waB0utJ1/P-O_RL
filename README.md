P-O_RL: PPO Locomotion for Unitree G1 in MuJoCo

一个基于 **MuJoCo + Ray 并行采样 + PPO** 的 Unitree G1 行走训练项目。  
目标是让 29 DoF 人形机器人在仿真中学会稳定前进步态。

## Features

- MuJoCo 物理仿真环境（`envs/mujoco_env.py`）
- Ray 多进程并行采样（`--num-procs`）
- PPO（支持 FF / LSTM policy）
- 可配置奖励项与系数（`envs/env_config.py` + `envs/reward.py`）
- TensorBoard 日志与 checkpoint 保存

## Project Structure

```text
esrl/
├── envs/
│   ├── env_config.py      # 环境/控制/奖励配置
│   ├── mujoco_env.py      # MuJoCo 环境主逻辑
│   ├── reward.py          # 奖励函数定义
│   └── robot_interface.py # 机器人状态与接触接口
├── rl/
│   ├── policies/          # actor / critic 网络
│   └── ppo/               # PPO 训练主流程与buffer
├── models/unitree_g1/     # MuJoCo XML 与网格
├── train.py               # 训练入口
└── run.py                 # 推理入口
```

## Requirements

- Python 3.10+（推荐）
- MuJoCo Python 包
- PyTorch
- Ray
- NumPy / SciPy
- TensorBoard

示例安装（仅供参考）：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch ray numpy scipy mujoco tensorboard
```

如果你在无显示环境（服务器）运行 MuJoCo，可能需要额外配置 EGL/OSMesa。

## Training

推荐从电子皮肤版本场景开始：

```bash
python train.py \
  --env models/unitree_g1/unitree_g1_es.xml \
  --num-procs 16 \
  --n-itr 500 \
  --max-traj-len 3000
```

训练输出默认保存在 `runs/train/exp*/`，包括：

- `actor.pt`, `critic.pt`（当前最优）
- `actor_<itr>.pt`, `critic_<itr>.pt`（定期评估快照）
- `experiment.pkl`（训练参数）
- `events.out.tfevents.*`（TensorBoard）

继续训练（resume）示例：

```bash
python train.py \
  --env models/unitree_g1/unitree_g1_es.xml \
  --continued runs/train/exp1/actor.pt \
  --num-procs 16 \
  --n-itr 300
```

## Evaluation / Play

```bash
python run.py \
  --env models/unitree_g1/unitree_g1_es.xml \
  --path runs/train/exp1
```

也可以直接传入 actor 文件路径：

```bash
python run.py \
  --env models/unitree_g1/unitree_g1_es.xml \
  --path runs/train/exp1/actor.pt
```

说明：

- `run.py` 会自动从同目录加载对应的 `critic*.pt`。
- 是否渲染由 `envs/env_config.py` 的 `G1Cfg.env.if_render` 控制（默认 `False`）。

## TensorBoard

```bash
tensorboard --logdir runs/train
```

## Key Configs

- 训练超参入口：`train.py`
- 奖励与控制参数：`envs/env_config.py`
- 奖励函数实现：`envs/reward.py`

建议先调这几项：

- `tracking_lin_vel`, `tracking_ang_vel`
- `orientation`, `base_height`, `termination`
- `action_rate`, `torques`
- `max-traj-len`, `num-procs`, `epochs`, `minibatch-size`

## Open-source Notes

- 本仓库已通过 `.gitignore` 忽略训练产物（如 `runs/`, `*.pt`, `*.pkl`）。
- 发布前请确认 `models/` 中第三方模型/网格的再分发许可。

## Known Limitations

- `run.py` 中 `--out-dir` / `--ep-len` 参数当前未实际用于导出视频。
- 不同机器上的 MuJoCo 图形后端配置可能不同，需要按环境调整。

