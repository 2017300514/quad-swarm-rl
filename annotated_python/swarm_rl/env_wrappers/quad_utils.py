# 中文注释副本；原始文件：swarm_rl/env_wrappers/quad_utils.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import copy

# 导入当前模块依赖。
import torch
from sample_factory.algo.learning.learner import Learner
from sample_factory.model.actor_critic import create_actor_critic

# 导入当前模块依赖。
from gym_art.quadrotor_multi.quad_experience_replay import ExperienceReplayWrapper
from swarm_rl.env_wrappers.compatibility import QuadEnvCompatibility
from swarm_rl.env_wrappers.reward_shaping import DEFAULT_QUAD_REWARD_SHAPING, QuadsRewardShapingWrapper
from swarm_rl.env_wrappers.v_value_map import V_ValueMapWrapper


# 定义类 `AnnealSchedule`。
class AnnealSchedule:
    # 定义函数 `__init__`。
    def __init__(self, coeff_name, final_value, anneal_env_steps):
        # 保存或更新 `coeff_name` 的值。
        self.coeff_name = coeff_name
        # 保存或更新 `final_value` 的值。
        self.final_value = final_value
        # 保存或更新 `anneal_env_steps` 的值。
        self.anneal_env_steps = anneal_env_steps


# 定义函数 `make_quadrotor_env_multi`。
def make_quadrotor_env_multi(cfg, render_mode=None, **kwargs):
    # 导入当前模块依赖。
    from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti
    # 保存或更新 `quad` 的值。
    quad = 'Crazyflie'
    # 保存或更新 `dyn_randomize_every` 的值。
    dyn_randomize_every = dyn_randomization_ratio = None
    # 保存或更新 `raw_control` 的值。
    raw_control = raw_control_zero_middle = True

    # 保存或更新 `sampler_1` 的值。
    sampler_1 = None
    # 根据条件决定是否进入当前分支。
    if dyn_randomization_ratio is not None:
        # 保存或更新 `sampler_1` 的值。
        sampler_1 = dict(type='RelativeSampler', noise_ratio=dyn_randomization_ratio, sampler='normal')

    # 保存或更新 `sense_noise` 的值。
    sense_noise = 'default'
    # 保存或更新 `dynamics_change` 的值。
    dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))

    # 保存或更新 `rew_coeff` 的值。
    rew_coeff = DEFAULT_QUAD_REWARD_SHAPING['quad_rewards']
    # 保存或更新 `use_replay_buffer` 的值。
    use_replay_buffer = cfg.replay_buffer_sample_prob > 0.0

    # 保存或更新 `env` 的值。
    env = QuadrotorEnvMulti(
        num_agents=cfg.quads_num_agents, ep_time=cfg.quads_episode_duration, rew_coeff=rew_coeff,
        obs_repr=cfg.quads_obs_repr,
        # Neighbor
        neighbor_visible_num=cfg.quads_neighbor_visible_num, neighbor_obs_type=cfg.quads_neighbor_obs_type,
        collision_hitbox_radius=cfg.quads_collision_hitbox_radius,
        collision_falloff_radius=cfg.quads_collision_falloff_radius,
        # Obstacle
        use_obstacles=cfg.quads_use_obstacles, obst_density=cfg.quads_obst_density, obst_size=cfg.quads_obst_size,
        obst_spawn_area=cfg.quads_obst_spawn_area,

        # Aerodynamics
        use_downwash=cfg.quads_use_downwash,
        # Numba Speed Up
        use_numba=cfg.quads_use_numba,
        # Scenarios
        quads_mode=cfg.quads_mode,
        # Room
        room_dims=cfg.quads_room_dims,
        # Replay Buffer
        use_replay_buffer=use_replay_buffer,
        # Rendering
        quads_view_mode=cfg.quads_view_mode, quads_render=cfg.quads_render,
        # Quadrotor Specific (Do Not Change)
        dynamics_params=quad, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
        dynamics_randomize_every=dyn_randomize_every, dynamics_change=dynamics_change, dyn_sampler_1=sampler_1,
        sense_noise=sense_noise, init_random_state=False,
        # Rendering
        render_mode=render_mode,
    )

    # 根据条件决定是否进入当前分支。
    if use_replay_buffer:
        # 保存或更新 `env` 的值。
        env = ExperienceReplayWrapper(env, cfg.replay_buffer_sample_prob, cfg.quads_obst_density, cfg.quads_obst_size,
                                      cfg.quads_domain_random, cfg.quads_obst_density_random, cfg.quads_obst_size_random,
                                      cfg.quads_obst_density_min, cfg.quads_obst_density_max, cfg.quads_obst_size_min, cfg.quads_obst_size_max)

    # 保存或更新 `reward_shaping` 的值。
    reward_shaping = copy.deepcopy(DEFAULT_QUAD_REWARD_SHAPING)

    # 保存或更新 `reward_shaping[quad_rewards][quadcol_bin]` 的值。
    reward_shaping['quad_rewards']['quadcol_bin'] = cfg.quads_collision_reward
    # 保存或更新 `reward_shaping[quad_rewards][quadcol_bin_smooth_max]` 的值。
    reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = cfg.quads_collision_smooth_max_penalty
    # 保存或更新 `reward_shaping[quad_rewards][quadcol_bin_obst]` 的值。
    reward_shaping['quad_rewards']['quadcol_bin_obst'] = cfg.quads_obst_collision_reward

    # this is annealed by the reward shaping wrapper
    # 根据条件决定是否进入当前分支。
    if cfg.anneal_collision_steps > 0:
        # 保存或更新 `reward_shaping[quad_rewards][quadcol_bin]` 的值。
        reward_shaping['quad_rewards']['quadcol_bin'] = 0.0
        # 保存或更新 `reward_shaping[quad_rewards][quadcol_bin_smooth_max]` 的值。
        reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = 0.0
        # 保存或更新 `reward_shaping[quad_rewards][quadcol_bin_obst]` 的值。
        reward_shaping['quad_rewards']['quadcol_bin_obst'] = 0.0
        # 保存或更新 `annealing` 的值。
        annealing = [
            AnnealSchedule('quadcol_bin', cfg.quads_collision_reward, cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_smooth_max', cfg.quads_collision_smooth_max_penalty,
                           cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_obst', cfg.quads_obst_collision_reward, cfg.anneal_collision_steps),
        ]
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 保存或更新 `annealing` 的值。
        annealing = None

    # 保存或更新 `env` 的值。
    env = QuadsRewardShapingWrapper(env, reward_shaping_scheme=reward_shaping, annealing=annealing,
                                    with_pbt=cfg.with_pbt)
    # 保存或更新 `env` 的值。
    env = QuadEnvCompatibility(env)

    # 根据条件决定是否进入当前分支。
    if cfg.visualize_v_value:
        # 保存或更新 `actor_critic` 的值。
        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
        # 调用 `eval` 执行当前处理。
        actor_critic.eval()

        # 执行这一行逻辑。
        device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
        # 调用 `model_to_device` 执行当前处理。
        actor_critic.model_to_device(device)

        # 保存或更新 `policy_id` 的值。
        policy_id = cfg.policy_index
        # 保存或更新 `name_prefix` 的值。
        name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
        # 保存或更新 `checkpoints` 的值。
        checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
        # 保存或更新 `checkpoint_dict` 的值。
        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        # 调用 `load_state_dict` 执行当前处理。
        actor_critic.load_state_dict(checkpoint_dict["model"])
        # 保存或更新 `env` 的值。
        env = V_ValueMapWrapper(env, actor_critic)

    # 返回当前函数的结果。
    return env


# 定义函数 `make_quadrotor_env`。
def make_quadrotor_env(env_name, cfg=None, _env_config=None, render_mode=None, **kwargs):
    # 根据条件决定是否进入当前分支。
    if env_name == 'quadrotor_multi':
        # 返回当前函数的结果。
        return make_quadrotor_env_multi(cfg, render_mode, **kwargs)
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 主动抛出异常以中止或提示错误。
        raise NotImplementedError
