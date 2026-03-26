#!/bin/bash

if [[ "${CONDA_DEFAULT_ENV:-}" != "marl" ]]; then
  source /home/i2r/miniconda3/etc/profile.d/conda.sh
  conda activate marl
fi

set -euo pipefail

env_name="mujoco"
scenario="Humanoid-v5"
algo="ppo"
seed=0
rollout_threads="${ROLLOUT_THREADS:-8}"
eval_threads="${EVAL_THREADS:-2}"
flow_samples="${FLOW_SAMPLES:-32}"
flow_steps="${FLOW_STEPS:-4}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "env=${env_name} scenario=${scenario} algo=${algo} seed=${seed} env_name=${CONDA_DEFAULT_ENV} rollout_threads=${rollout_threads} flow_samples=${flow_samples} flow_steps=${flow_steps}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python ppo/scripts/train/train_mujoco.py \
  --seed "${seed}" \
  --env_name "${env_name}" \
  --algorithm_name "${algo}" \
  --experiment_name "flow_ppo" \
  --scenario "${scenario}" \
  --value_model_type flow \
  --flow_base_dist normal \
  --flow_num_samples "${flow_samples}" \
  --flow_solver_steps "${flow_steps}" \
  --flow_matching_loss_coef 1.0 \
  --flow_endpoint_loss_coef 0.25 \
  --flow_monotonicity_coef 0.1 \
  --flow_target_ema 0.995 \
  --adv_metric_type transport_jacobian \
  --adv_expansion_coef 0.25 \
  --adv_magnitude_coef 0.1 \
  --critic_lr 3e-5 \
  --lr 3e-5 \
  --entropy_coef 0.01 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --max_grad_norm 0.8 \
  --n_training_threads 8 \
  --n_rollout_threads "${rollout_threads}" \
  --n_eval_rollout_threads "${eval_threads}" \
  --num_mini_batch 4 \
  --episode_length 500 \
  --eval_interval 25 \
  --eval_episodes 2 \
  --num_env_steps 200000000 \
  --ppo_epoch 20 \
  --clip_param 0.1 \
  --num_quants 64 \
  --use_eval \
  --use_value_active_masks \
  --use_policy_active_masks
