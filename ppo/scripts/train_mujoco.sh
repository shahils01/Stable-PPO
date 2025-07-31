#!/bin/sh
env="mujoco"
scenario="Humanoid-v4"
algo="ppo"
seed=0

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"
python train/train_mujoco.py --seed ${seed} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --critic_lr 3e-5 --lr 3e-5 --entropy_coef 0.001 --gamma 0.99 --gae_lambda 0.95 --max_grad_norm 0.8 --eval_episodes 2 --n_training_threads 32 --n_rollout_threads 64 --num_mini_batch 64 --episode_length 2048 --eval_interval 25 --num_env_steps 200000000 --ppo_epoch 20 --clip_param 0.1 --add_center_xy --use_state_agent --use_value_active_masks --use_policy_active_masks --use_wandb True --wandb_name "xxx" --user_name "shahil-shaik7-clemson-university"