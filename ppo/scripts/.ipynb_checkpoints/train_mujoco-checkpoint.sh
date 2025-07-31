#!/bin/sh
env="mujoco"
scenario="HalfCheetah-v5"
agent_conf="6x1"
agent_obsk=0
faulty_node=-1
#eval_faulty_node="-1 0 1 2 3 4 5"
eval_faulty_node="-1"
algo="mat_gnn"
exp="single"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python train/train_mujoco.py --seed ${seed} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --agent_conf ${agent_conf} --agent_obsk ${agent_obsk} --faulty_node ${faulty_node} --eval_faulty_node ${eval_faulty_node} --critic_lr 2.0633e-05 --lr 2.0633e-05 --entropy_coef 0.000401762 --max_grad_norm 0.8 --eval_episodes 2 --n_training_threads 16 --n_rollout_threads 64 --num_mini_batch 40 --episode_length 512 --eval_interval 25 --num_env_steps 200000000 --ppo_epoch 20 --clip_param 0.1 --use_eval --add_center_xy --use_state_agent --use_value_active_masks --use_policy_active_masks --n_quants 64 --iterations 6 --num-heads 1 --num-layers 2 --use_wandb True --wandb_name "xxx" --user_name "shahil-shaik7-clemson-university"