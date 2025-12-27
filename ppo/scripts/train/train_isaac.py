"""
Script to train your Custom PPO on Isaac Lab environments.
"""
import argparse
import sys
import os
import socket
import wandb
import setproctitle
from pathlib import Path

# --- Isaac Lab Launch ---
from isaaclab.app import AppLauncher

# 1. SETUP FIRST PARSER (Isaac Lab specific)
parser = argparse.ArgumentParser(description="Train custom PPO on Isaac Lab.")
parser.add_argument("--task", type=str, default="Isaac-Ant-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")

# Add AppLauncher args (headless, etc.)
AppLauncher.add_app_launcher_args(parser)

# 2. CHANGE THIS: Use parse_known_args instead of parse_args
# args_cli will hold Isaac args, extras will hold your PPO args (lr, gamma, etc.)
args_cli, extras = parser.parse_known_args()

# Launch the app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports after launch ---
import gymnasium as gym
import torch
import isaaclab.envs
from isaaclab_tasks.utils import parse_env_cfg

sys.path.append("../../")
from ppo.config import get_config
from ppo.envs.isaac_wrapper import IsaacLabVecEnvWrapper
from ppo.runner.shared.isaac_runner import IsaacRunner as Runner

# 3. UPDATE PARSE_ARGS TO ACCEPT THE 'EXTRAS' LIST
def parse_ppo_args(extra_args, parser):
    parser.add_argument('--scenario', type=str, default='Isaac-Ant-v0', help="Which task to run on")
    # Parse the remaining arguments known to the PPO config
    all_args = parser.parse_known_args(extra_args)[0]
    return all_args

def main(ppo_args_list):
    # 0. Load Custom DPPO Config
    parser = get_config()
    
    # Pass the 'extras' list we caught earlier
    all_args = parse_ppo_args(ppo_args_list, parser)
    
    print("mumu config: ", all_args)

    # 1. Configure the Environment
    # Note: You were using args_cli.task in gym.make, but all_args.scenario in parse_env_cfg.
    # It is safer to use one source of truth.
    task_name = all_args.scenario if all_args.scenario else args_cli.task
    
    # Training env config
    env_cfg = parse_env_cfg(task_name, num_envs=all_args.n_rollout_threads)
    
    # 2. Create the Isaac Lab Environment
    env = gym.make(task_name, cfg=env_cfg)
    
    # 3. Wrap it for your PPO
    vec_env = IsaacLabVecEnvWrapper(env)

    # Create run directory
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.scenario / all_args.algorithm_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.scenario,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_seed_" + str(all_args.seed),
                         group='DistPPO',
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            curr_run = 'run1' if len(exst_run_nums) == 0 else 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # 4. Setup Your Config
    config = {
        "all_args": all_args,
        "envs": vec_env,
        "eval_envs": vec_env,
        "device": torch.device("cuda:0"),
        "run_dir": run_dir
    }

    # 5. Initialize and Run
    runner = Runner(config)
    runner.run()

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    # Pass the extras found at the top of the script
    main(extras)