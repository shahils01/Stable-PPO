#!/bin/sh
env="StarCraft2"
map="5m_vs_6m"    # 5m_vs_6m, MMM2
algo="mat_gnn"
exp="single"
seed=4

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --eval_map_name ${map} --seed ${seed} --n_training_threads 16 --n_rollout_threads 32 --num_mini_batch 1 --episode_length 300 --num_env_steps 20000000 --lr 5e-4 --ppo_epoch 20 --gamma 0.98 --gae_lambda 0.92 --clip_param 0.05 --save_interval 100000 --use_value_active_masks --entropy_coef 0.001 --max_grad_norm 0.5 --encode_state True --n_quants 1 --iterations 64 --hidden_size 512 --hid-dim 512 --n_embd 512 --out_channels 512 --use_eval --use_wandb False --wandb_name "xxx" --user_name "shahil-shaik7-clemson-university"

# If smac fails, enter the command: pkill -f "SC2_x64 -listen"
# CUDA_VISIBLE_DEVICES=0 
