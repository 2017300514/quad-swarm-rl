# Hybrid Auto Reproduction Notes

## 1. Workspace choice

For reproducing the obstacle-avoidance paper described by:

- `~/sui_work_not_delete/quad-swarm-rl-master/hybrid_auto/Huang 等 - Collision Avoidance and Navigation for a Quadrotor.md`

the better code workspace is:

- `/home/server2/sui_work_not_delete/quad-swarm-rl`

### Why this workspace is the safer base

1. It is on the upstream-style `master` branch and matches the published repository layout more closely.
2. It already contains the obstacle training entry points used by the paper family:
   - `swarm_rl/runs/obstacles/quad_obstacle_baseline.py`
   - `swarm_rl/runs/obstacles/quads_multi_obstacles.py`
   - `swarm_rl/runs/obstacles/pbt_quads_multi_obstacles.py`
3. The competing workspace `~/sui_work_not_delete/quad-swarm-rl-master` contains local, uncommitted modifications:
   - `hybrid_auto/`
   - `setup_swarm_rl_env_no_torch.sh`
   - debug-oriented changes in `swarm_rl/train.py`
   - comment-heavy edits in `swarm_rl/models/quad_multi_model.py`
   - extra local files such as `swarm_rl/debug_utils.py`
4. Those local edits do not add clear paper functionality. They mainly add comments and debug hooks, which increases drift from the upstream code path and therefore increases reproduction risk.

### Practical conclusion

Use:

- code base: `/home/server2/sui_work_not_delete/quad-swarm-rl`
- paper notes and extracted materials: `~/sui_work_not_delete/quad-swarm-rl-master/hybrid_auto`

That gives you a cleaner training codebase while keeping the paper artifacts available for reference.

## 2. Paper-related code entry points

The obstacle-navigation paper content aligns with the obstacle training configuration in:

- `/home/server2/sui_work_not_delete/quad-swarm-rl/swarm_rl/runs/obstacles/quad_obstacle_baseline.py`
- `/home/server2/sui_work_not_delete/quad-swarm-rl/swarm_rl/runs/obstacles/pbt_quads_multi_obstacles.py`

Important defaults visible in the code:

- `--quads_use_obstacles=True`
- `--quads_obstacle_obs_type=octomap`
- `--quads_obst_density=0.2`
- `--quads_obst_size=0.6`
- `--replay_buffer_sample_prob=0.75`
- `--quads_encoder_type=attention`
- `--quads_neighbor_obs_type=pos_vel`

These are consistent with the paper themes: obstacle observations, replay support, and attention-based interaction modeling.

## 3. Python environment choice

Use:

- Python `3.11.11`
- Conda environment name suggestion: `swarm-rl-obstacles`

Reasoning:

1. The existing local reference script already targets Python `3.11.11`.
2. `setup.py` requires `python>=3.11.10`.
3. This avoids trying to run the project on the host default Python `3.13.11`, which is much more likely to break older RL dependencies.

## 4. PyTorch choice

Observed host GPU state:

- 4 x NVIDIA GeForce RTX 4090
- Driver Version: `590.48.01`
- Reported CUDA Version: `13.1`

Recommended PyTorch install:

- `torch==2.5.0`
- CUDA wheel channel: `cu124`

Reasoning:

1. This repository pins `torch==2.5.0` in `setup.py`.
2. A `cu124` wheel is a stable match for modern NVIDIA drivers and is the safest choice for this pinned torch version.
3. You do not need a local CUDA toolkit install for pip wheels; the NVIDIA driver is what matters for runtime.

## 5. Environment creation script

Created script:

- `/home/server2/sui_work_not_delete/quad-swarm-rl/setup_swarm_rl_env_with_torch.sh`

Run:

```bash
cd /home/server2/sui_work_not_delete/quad-swarm-rl
bash setup_swarm_rl_env_with_torch.sh
```

Custom environment name:

```bash
bash setup_swarm_rl_env_with_torch.sh swarm-rl-paper
```

## 6. Validation commands

After installation:

```bash
conda activate swarm-rl-obstacles
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -m swarm_rl.train --env=quadrotor_multi --help
./run_tests.sh
```

## 7. Operational cautions

1. At the time of inspection, all 4 GPUs already had active Python workloads attached.
2. Start training only after selecting an idle GPU, for example with `CUDA_VISIBLE_DEVICES`.
3. Rendering tests may require an X11 display.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m swarm_rl.train --env=quadrotor_multi --help
```

## 8. Commands that need your authorization

No system-level package installation is required for the current plan.

Authorization is only needed if you want actions such as:

1. installing or changing OS packages with `apt`, `yum`, or similar
2. changing NVIDIA drivers, CUDA toolkit, or host-wide libraries
3. modifying files outside the writable project/user scope

The environment script only creates a user-space conda environment and installs Python packages into it.
