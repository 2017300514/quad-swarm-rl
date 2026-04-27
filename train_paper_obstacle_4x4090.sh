#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-swarm-rl-obstacles}"
TRAIN_ROOT="${TRAIN_ROOT:-$ROOT_DIR/train_dir/paper_huang_obstacle/final}"
LOG_DIR="${LOG_DIR:-$TRAIN_ROOT/launcher_logs}"

# Paper-faithful defaults are heavy on CPU. Override if needed, e.g.
# NUM_WORKERS_PER_SEED=12 NUM_ENVS_PER_WORKER=4 bash train_paper_obstacle_4x4090.sh
NUM_WORKERS_PER_SEED="${NUM_WORKERS_PER_SEED:-36}"
NUM_ENVS_PER_WORKER="${NUM_ENVS_PER_WORKER:-4}"
TRAIN_FOR_ENV_STEPS="${TRAIN_FOR_ENV_STEPS:-1000000000}"

GPUS=(0 1 2 3)
SEEDS=(0000 1111 2222 3333)

if [ "${#GPUS[@]}" -ne "${#SEEDS[@]}" ]; then
  echo "GPU/seed count mismatch" >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    CONDA_BASE="$(conda info --base)"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
  fi
  conda activate "$CONDA_ENV"
fi

mkdir -p "$TRAIN_ROOT" "$LOG_DIR"

COMMON_ARGS=(
  --env=quadrotor_multi
  --algo=APPO
  --device=gpu
  --train_for_env_steps="$TRAIN_FOR_ENV_STEPS"
  --use_rnn=False
  --num_workers="$NUM_WORKERS_PER_SEED"
  --num_envs_per_worker="$NUM_ENVS_PER_WORKER"
  --learning_rate=0.0001
  --ppo_clip_value=5.0
  --recurrence=1
  --nonlinearity=tanh
  --actor_critic_share_weights=False
  --policy_initialization=xavier_uniform
  --adaptive_stddev=False
  --with_vtrace=False
  --max_policy_lag=100000000
  --rnn_size=256
  --gae_lambda=1.00
  --max_grad_norm=5.0
  --exploration_loss_coeff=0.0
  --rollout=128
  --batch_size=1024
  --with_pbt=False
  --normalize_input=False
  --normalize_returns=False
  --reward_clip=10
  --quads_use_numba=True
  --save_milestones_sec=3600
  --anneal_collision_steps=300000000
  --replay_buffer_sample_prob=0.75
  --quads_mode=mix
  --quads_episode_duration=15.0
  --quads_obs_repr=xyz_vxyz_R_omega_floor
  --quads_neighbor_hidden_size=256
  --quads_neighbor_obs_type=pos_vel
  --quads_collision_hitbox_radius=2.0
  --quads_collision_falloff_radius=4.0
  --quads_collision_reward=5.0
  --quads_collision_smooth_max_penalty=4.0
  --quads_neighbor_encoder_type=no_encoder
  --quads_neighbor_visible_num=2
  --quads_use_obstacles=True
  --quads_obst_spawn_area 8 8
  --quads_obst_density=0.2
  --quads_obst_size=0.6
  --quads_obst_collision_reward=5.0
  --quads_obstacle_obs_type=octomap
  --quads_use_downwash=True
  --quads_num_agents=8
  --quads_encoder_type=attention
  --train_dir="$TRAIN_ROOT"
)

PIDS=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}

trap cleanup INT TERM

echo "Launching 4 paper-style training jobs"
echo "  train_root: $TRAIN_ROOT"
echo "  log_dir:    $LOG_DIR"
echo "  workers:    $NUM_WORKERS_PER_SEED"
echo "  envs/worker:$NUM_ENVS_PER_WORKER"

for idx in "${!SEEDS[@]}"; do
  gpu="${GPUS[$idx]}"
  seed="${SEEDS[$idx]}"
  experiment="seed_${seed}"
  log_file="$LOG_DIR/${experiment}.log"

  echo "Starting ${experiment} on GPU ${gpu} -> ${log_file}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    cd "$ROOT_DIR"
    python -m swarm_rl.train \
      "${COMMON_ARGS[@]}" \
      --seed="$seed" \
      --experiment="$experiment"
  ) >"$log_file" 2>&1 &

  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo "All four seed jobs exited."
