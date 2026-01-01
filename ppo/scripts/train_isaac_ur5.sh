#!/bin/bash

# --- Configuration Variables ---
SIF_FILE="/home/shahils/isaacLab_Apptainer/isaac_lab.sif"
PROJECT_PATH="/home/shahils/Desktop/gitBackupRepo/Stable-PPO/ppo/isaac_files/isaac_env_ur5/RL_UR5_IsaacLab"
PACKAGE_PATH="${PROJECT_PATH}/source/RL_UR5"
# The CUDA device to use
export CUDA_VISIBLE_DEVICES=0
# Run in headless mode (required for servers)
export HEADLESS=1

env="IsaacLab"

# --- Volume Bindings for Apptainer ---
# These paths map host directories to container directories for persistence and access
BINDINGS="-B /home/shahils/Desktop/gitBackupRepo/Stable-PPO/ppo/isaac_files/isaac-sim/kit_data:/isaac-sim/kit/data"
BINDINGS="$BINDINGS -B /home/shahils/Desktop/gitBackupRepo/Stable-PPO/ppo/isaac_files/isaac-sim/kit_cache:/isaac-sim/kit/cache"
BINDINGS="$BINDINGS -B /home/shahils/Desktop/gitBackupRepo/Stable-PPO/ppo/isaac_files/isaac-sim/ov_data:$HOME/.local/share/ov/data "
BINDINGS="$BINDINGS -B /home/shahils/Desktop/gitBackupRepo/Stable-PPO/ppo/isaac_files/isaac-sim/logs:$HOME/.nvidia-omniverse/logs"
BINDINGS="$BINDINGS -B /home/shahils/Desktop/gitBackupRepo/Stable-PPO/ppo/isaac_files/isaac-sim/documents:$HOME/Documents"
# CRITICAL: Bind your project path so the container can access the code
BINDINGS="$BINDINGS -B ${PROJECT_PATH}:${PROJECT_PATH}"

# --- Execution ---

echo "Starting Isaac Lab container and installing custom package..."

# The command to execute inside the container is passed as a string to the shell.
# We use a semi-colon (;) to separate the three steps:
# 1. Install the local package in editable mode (`-e`)
# 2. Run the isaaclab.sh wrapper with -p (for pip) and -m (for module)
# 3. Execute the training script
apptainer exec --nv ${BINDINGS} ${SIF_FILE} /bin/bash -c " \
    # 1. Install the custom Python package in 'editable' mode
    # Note: Use the python executable specific to Isaac Lab
    /opt/IsaacLab/isaaclab.sh -p -m pip install -e ${PACKAGE_PATH}; \
    \
    # 2. Run the training command through the isaaclab wrapper script
    # The wrapper sets up the environment correctly before executing the script
    /opt/IsaacLab/isaaclab.sh -p train/train_isaac_ur5.py \
      --env_name ${env} \
      --scenario Isaac-UR5-HuberDirectObj-PPO \
      --num_envs 32 \
      --enable_cameras \
      --use_image \
      --episode_length 200 \
      --num_quants 64 \
      --n_training_threads 32 \
      --num_mini_batch 4 \
      --num_env_steps 200000000 \
      --use_wandb True \
      --wandb_name "xxx" \
      --user_name "shahil-shaik7-clemson-university" \
      --headless; \
"
echo "Training command finished."