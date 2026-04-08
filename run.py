import torch
import pickle
import argparse
import numpy as np
from pathlib import Path
from functools import partial
from envs.mujoco_env import MujocoEnv as Env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", required=True, type=str)
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--out-dir", required=False, type=Path, default=None, help="Path to directory to save videos")
    parser.add_argument("--ep-len", required=False, type=int, default=10, help="Episode length to play (in seconds)")

    args = parser.parse_args()

    path_to_actor = ""
    if args.path.is_file() and args.path.suffix==".pt":
        path_to_actor = args.path
    elif args.path.is_dir():
        path_to_actor = Path(args.path, "actor.pt")
    else:
        raise Exception("Invalid path to actor module: ", args.path)

    path_to_critic = Path(path_to_actor.parent, "critic" + str(path_to_actor).split('actor')[1])
    path_to_pkl = Path(path_to_actor.parent, "experiment.pkl")

    # load trained policy
    actor = torch.load(path_to_actor, weights_only=False)
    critic = torch.load(path_to_critic, weights_only=False)
    actor.eval()
    critic.eval()

    env = Env(args.env)

    observation = env.reset()

    done = False

    # 隐藏状态初始化
    if hasattr(actor, 'init_hidden_state'): 
        actor.init_hidden_state()
    if hasattr(critic, 'init_hidden_state'):
        critic.init_hidden_state()

    if_over = False
    i = 0
    #while not done:
    #while True:
    while not if_over:
        action = actor.forward(torch.tensor(observation, dtype=torch.float32), deterministic=True).detach().numpy()
        observation, reward, done, _ = env.step(action.copy())

        i += 1
        if i >= 10000:
            if_over = True
            print("时间差不多咯")