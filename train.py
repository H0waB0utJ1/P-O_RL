import re
import ray
import pickle
import argparse
from pathlib import Path
from functools import partial

from rl.ppo.ppo import PPO
from envs.mujoco_env import MujocoEnv as Env

def run_experiment(args):
    # wrapper function for creating parallelized envs
    env_fn = partial(Env, xml_path=args.env)

    # Set up Parallelism
    if not ray.is_initialized():
        ray.init(num_cpus=args.num_procs)

    # dump hyperparameters
    base_logdir = Path(args.logdir)
    base_logdir.mkdir(parents=True, exist_ok=True)
    existing = [d for d in base_logdir.iterdir() if d.is_dir() and re.match(r"exp\d+", d.name)]
    if existing:
        ids = [int(d.name[3:]) for d in existing]
        next_id = max(ids) + 1
    else:
        next_id = 1
    exp_logdir = base_logdir / f"exp{next_id}"
    exp_logdir.mkdir(parents=True, exist_ok=True)
    args.logdir = exp_logdir
    pkl_path = exp_logdir / "experiment.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(args, f)
    print(f"Logging to: {exp_logdir}")

    algo = PPO(env_fn, args)
    algo.train(env_fn, args.n_itr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", required=True, type=str)
    parser.add_argument("--logdir", default=Path("runs/train"), type=Path, help="Path to save weights and logs")
    parser.add_argument("--n-itr", type=int, default=500, help="Number of iterations of the learning algorithm")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate") # Xie
    parser.add_argument("--eps", type=float, default=1e-5, help="Adam epsilon (for numerical stability)")
    parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
    parser.add_argument("--gamma", type=float, default=0.995, help="MDP discount")
    parser.add_argument("--std-dev", type=float, default=0.5, help="Action noise for exploration")
    parser.add_argument("--learn-std", action="store_true", help="Exploration noise will be learned")
    parser.add_argument("--entropy-coeff", type=float, default=0.001, help="Coefficient for entropy regularization")
    parser.add_argument("--clip", type=float, default=0.2, help="Clipping parameter for PPO surrogate loss")
    parser.add_argument("--minibatch-size", type=int, default=32, help="Batch size for PPO updates")
    parser.add_argument("--epochs", type=int, default=4, help="Number of optimization epochs per PPO update")
    parser.add_argument("--num-procs", type=int, default=16, help="Number of threads to train on")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Value to clip gradients at")
    parser.add_argument("--max-traj-len", type=int, default=3000, help="Max episode horizon")
    parser.add_argument("--max-n-traj", type=int, default=4, help="Max traj number")
    parser.add_argument("--eval-freq", required=False, default=50, type=int, help="Frequency of performing evaluation")
    parser.add_argument("--continued", required=False, type=Path, help="path to pretrained weights")
    parser.add_argument("--recurrent", required=False, action="store_true", help="use LSTM instead of FF")

    args = parser.parse_args()

    run_experiment(args)
