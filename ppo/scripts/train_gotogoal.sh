#!/bin/sh
env="mujoco"
scenario="GoToGoal-v0"
algo="ppo"
seed=0

# echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"
python train/train_gotogoal.py \
 --seed ${seed} \
 --env_name ${env} \
 --algorithm_name ${algo} \
 --experiment_name ${exp} \
 --scenario ${scenario} \
 --model_dir "/home/yue6/shahil_ws/Stable-PPO/ppo/scripts/results/mujoco/GoToGoal-v0/ppo/wandb/run-20260119_111344-diq56rgs/files/transformer_200.pt" \
 --use_value_entropy True \
 --true_integration True \
 --critic_lr 5e-4 \
 --lr 5e-4 \
 --entropy_coef 0.01 \
 --gamma 0.99 \
 --gae_lambda 0.95 \
 --max_grad_norm 0.8 \
 --eval_episodes 2 \
 --n_training_threads 32 \
 --n_rollout_threads 128 \
 --num_mini_batch 1 \
 --episode_length 100 \
 --eval_interval 25 \
 --num_env_steps 200000000 \
 --ppo_epoch 20 \
 --clip_param 0.2 \
 --use_eval \
 --add_center_xy \
 --use_state_agent \
 --use_value_active_masks \
 --use_policy_active_masks \
 --num_quants 1 \
 --use_wandb True \
 --wandb_name "xxx" \
 --user_name "shahil-shaik7-clemson-university" \
#  --moe_policy True