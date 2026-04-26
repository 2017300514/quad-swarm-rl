# 中文注释副本；原始文件：swarm_rl/env_wrappers/quad_utils.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件是“配置对象 -> 可训练环境实例”之间的接线层。
# 它把命令行和实验配置里的选项，翻译成 `QuadrotorEnvMulti` 的构造参数，
# 然后按需要依次挂上 replay、reward shaping、兼容层和 V-value 可视化包装器。
# 上游输入是训练脚本解析好的 `cfg`；下游输出是 Sample Factory 可以直接采样的环境对象。

import copy

import torch
from sample_factory.algo.learning.learner import Learner
from sample_factory.model.actor_critic import create_actor_critic

from gym_art.quadrotor_multi.quad_experience_replay import ExperienceReplayWrapper
from swarm_rl.env_wrappers.compatibility import QuadEnvCompatibility
from swarm_rl.env_wrappers.reward_shaping import DEFAULT_QUAD_REWARD_SHAPING, QuadsRewardShapingWrapper
from swarm_rl.env_wrappers.v_value_map import V_ValueMapWrapper


class AnnealSchedule:
    # 这个小对象只保存奖励退火所需的三元信息：
    # 哪个奖励系数要退火、目标终值是多少、以及在多少环境步内完成退火。
    # 它本身不执行更新，真正消费这些字段的是 `QuadsRewardShapingWrapper.step()`。
    def __init__(self, coeff_name, final_value, anneal_env_steps):
        self.coeff_name = coeff_name
        self.final_value = final_value
        self.anneal_env_steps = anneal_env_steps


def make_quadrotor_env_multi(cfg, render_mode=None, **kwargs):
    # 这里延迟导入环境实现，避免在仅做参数解析或模块注册时提前拉起较重的环境依赖。
    from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti

    # 这一组常量决定当前训练默认采用哪种飞行器动力学配置和控制接口。
    # `Crazyflie` 指向项目里预设的小型四旋翼动力学参数；
    # `raw_control=True` 表示策略直接输出更底层的控制量，而不是更高层的 waypoint/速度命令。
    quad = 'Crazyflie'
    dyn_randomize_every = dyn_randomization_ratio = None
    raw_control = raw_control_zero_middle = True

    # 如果启用动力学随机化，这里会构造参数采样器，让不同 episode 在质量、推力等物理参数上产生扰动。
    # 当前默认是 `None`，说明主流程里这条随机化链路没有打开。
    sampler_1 = None
    if dyn_randomization_ratio is not None:
        sampler_1 = dict(type='RelativeSampler', noise_ratio=dyn_randomization_ratio, sampler='normal')

    # 传感噪声和动力学修正项会在单机环境内部继续细化。
    # 这里给出的 `dynamics_change` 不是奖励参数，而是施加到底层物理模型上的扰动配置。
    sense_noise = 'default'
    dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))

    # 这一份基础奖励表来自 reward shaping 模块的默认配置。
    # 它先作为环境内部 `rew_coeff` 的初始值写进 `QuadrotorEnvMulti`，
    # 后面再根据命令行参数覆盖碰撞相关项，并由 wrapper 决定是否做退火。
    rew_coeff = DEFAULT_QUAD_REWARD_SHAPING['quad_rewards']

    # replay 是否启用并不看单独布尔开关，而是由采样概率是否大于 0 决定。
    # 这样实验配置只需要控制“多大概率从碰撞片段重开”，不必再维护第二个同步开关。
    use_replay_buffer = cfg.replay_buffer_sample_prob > 0.0

    # 这里是环境主实例化步骤。
    # 上面在 `quadrotor_params.py` 里登记的参数，在这里第一次真正落到多机环境构造函数中：
    # agent 数量、观测形式、邻居设置、障碍物、房间尺寸、下洗气流、渲染与回放开关都从 `cfg` 流入环境。
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

    # 如果启用 replay，这里再往环境外层包一层碰撞片段恢复逻辑。
    # 注意这一层拿到的不只是采样概率，还拿到障碍密度/尺寸及其随机化范围，
    # 因为 replay 恢复出来的场景必须和原来触发碰撞时的环境条件保持一致或可控地随机化。
    if use_replay_buffer:
        env = ExperienceReplayWrapper(env, cfg.replay_buffer_sample_prob, cfg.quads_obst_density, cfg.quads_obst_size,
                                      cfg.quads_domain_random, cfg.quads_obst_density_random, cfg.quads_obst_size_random,
                                      cfg.quads_obst_density_min, cfg.quads_obst_density_max, cfg.quads_obst_size_min, cfg.quads_obst_size_max)

    # 这里单独拷贝一份 reward shaping 配置，而不是直接改全局默认表，
    # 是为了避免不同实验或不同环境实例之间共享可变字典，导致一次修改污染其他训练任务。
    reward_shaping = copy.deepcopy(DEFAULT_QUAD_REWARD_SHAPING)

    # 这三项把命令行里的碰撞相关系数写回本次实验专属的奖励表：
    # 机间硬碰撞、连续接近惩罚、障碍碰撞惩罚。
    reward_shaping['quad_rewards']['quadcol_bin'] = cfg.quads_collision_reward
    reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = cfg.quads_collision_smooth_max_penalty
    reward_shaping['quad_rewards']['quadcol_bin_obst'] = cfg.quads_obst_collision_reward

    # 如果启用了退火，训练初期先把碰撞相关权重清零，
    # 让策略先学会最基本的飞行与到达目标，再随着训练步数增加逐步强化安全约束。
    # 真正的线性增长逻辑不在这里执行，而是交给 `QuadsRewardShapingWrapper` 在 episode 结束时按训练步更新。
    if cfg.anneal_collision_steps > 0:
        reward_shaping['quad_rewards']['quadcol_bin'] = 0.0
        reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = 0.0
        reward_shaping['quad_rewards']['quadcol_bin_obst'] = 0.0
        annealing = [
            AnnealSchedule('quadcol_bin', cfg.quads_collision_reward, cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_smooth_max', cfg.quads_collision_smooth_max_penalty,
                           cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_obst', cfg.quads_obst_collision_reward, cfg.anneal_collision_steps),
        ]
    else:
        annealing = None

    # 这一层 wrapper 负责两件事：
    # 1. 把上面整理好的奖励系数真正写进环境的 `rew_coeff`
    # 2. 记录 episode 级奖励统计，并在需要时执行碰撞惩罚退火
    env = QuadsRewardShapingWrapper(env, reward_shaping_scheme=reward_shaping, annealing=annealing,
                                    with_pbt=cfg.with_pbt)

    # 兼容层的作用是把环境输出对齐到 Sample Factory 期望的接口形式，
    # 这样后面的采样与训练循环不需要知道底层四旋翼环境有哪些项目特有细节。
    env = QuadEnvCompatibility(env)

    # 这个分支不服务于普通训练，而是服务于训练后价值函数可视化。
    # 它会额外构造 actor-critic、加载 checkpoint，并把模型挂进 `V_ValueMapWrapper`，
    # 让环境在渲染或分析阶段能够查询当前状态附近的 value 分布。
    if cfg.visualize_v_value:
        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
        actor_critic.eval()

        device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
        actor_critic.model_to_device(device)

        policy_id = cfg.policy_index
        name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict["model"])
        env = V_ValueMapWrapper(env, actor_critic)

    return env


def make_quadrotor_env(env_name, cfg=None, _env_config=None, render_mode=None, **kwargs):
    # 这里是注册给 Sample Factory 的统一环境工厂入口。
    # 当前项目只支持 `quadrotor_multi`，所以这里实际是在做一个名字到具体构造函数的分发。
    if env_name == 'quadrotor_multi':
        return make_quadrotor_env_multi(cfg, render_mode, **kwargs)
    else:
        raise NotImplementedError
