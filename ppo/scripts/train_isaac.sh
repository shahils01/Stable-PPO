#!/bin/sh
env="IsaacLab"
scenario="offroad_jackal_RL_v0"  #"Isaac-Ant-v0"
algo="ppo"
seed=3

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"
# CUDA_VISIBLE_DEVICES=0 HEADLESS=1 apptainer exec --nv \
#   -B /home/shahils/Desktop/gitBackupRepo/Stable-PPO/ppo/isaac_files/isaac-sim/kit_data:/isaac-sim/kit/data \
#   -B /home/shahils/Desktop/gitBackupRepo/Stable-PPO/ppo/isaac_files/isaac-sim/kit_cache:/isaac-sim/kit/cache \
#   -B /home/shahils/Desktop/gitBackupRepo/Stable-PPO/ppo/isaac_files/isaac-sim/ov_data:$HOME/.local/share/ov/data \
#   -B /home/shahils/Desktop/gitBackupRepo/Stable-PPO/ppo/isaac_files/isaac-sim/logs:$HOME/.nvidia-omniverse/logs \
#   -B /home/shahils/Desktop/gitBackupRepo/Stable-PPO/ppo/isaac_files/isaac-sim/documents:$HOME/Documents \
#  /home/shahils/isaacLab_Apptainer/isaac_lab.sif \
#  /opt/IsaacLab/isaaclab.sh -p train/train_isaac.py \
python3 train/train_isaac.py \
 --seed ${seed} \
 --env_name ${env} \
 --algorithm_name ${algo} \
 --experiment_name ${exp} \
 --scenario ${scenario} \
 --critic_lr 3e-4 \
 --lr 3e-4 \
 --num_quants 1 \
 --entropy_coef 0.0 \
 --gamma 0.99 \
 --gae_lambda 0.95 \
 --max_grad_norm 1.0 \
 --n_training_threads 32 \
 --n_rollout_threads 1 \
 --num_mini_batch 4 \
 --episode_length 1000 \
 --eval_interval 25 \
 --num_env_steps 200000000 \
 --ppo_epoch 8 \
 --clip_param 0.2 \
 --add_center_xy \
 --use_state_agent \
 --use_value_active_masks \
 --use_policy_active_masks \
#  --use_wandb True \
#  --wandb_name "xxx" \
#  --user_name "shahil-shaik7-clemson-university" \
#  --moe_policy True \
#  --num_experts 4 \


# Do the following when running on Apptainer:
# mkdir -p /$path-to-your-repo$/isaac-files
# cd /$path-to-your-repo$/isaac-files
# mkdir -p /isaac-sim/kit_data
# mkdir -p /isaac-sim/kit_cache
# mkdir -p /isaac-sim/ov_data
# mkdir -p /isaac-sim/logs
# mkdir -p /isaac-sim/documents

# Then execute apptainer with the following:
# CUDA_VISIBLE_DEVICES=0 HEADLESS=1 apptainer exec --nv \
#   -B /$path-to-your-repo$/isaac_files/isaac-sim/kit_data:/isaac-sim/kit/data \
#   -B /$path-to-your-repo$/isaac_files/isaac-sim/kit_cache:/isaac-sim/kit/cache \
#   -B /$path-to-your-repo$/isaac_files/isaac-sim/ov_data:$HOME/.local/share/ov/data \
#   -B /$path-to-your-repo$/isaac_files/isaac-sim/logs:$HOME/.nvidia-omniverse/logs \
#   -B /$path-to-your-repo$/isaac_files/isaac-sim/documents:$HOME/Documents \
#  /home/shahils/isaacLab_Apptainer/isaac_lab.sif \
