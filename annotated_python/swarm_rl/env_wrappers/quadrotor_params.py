# 中文注释副本；原始文件：swarm_rl/env_wrappers/quadrotor_params.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件是四旋翼项目的参数入口面板。
# 它不直接执行训练或仿真，而是把“多机数量、观测形式、碰撞惩罚、障碍物设置、场景类型、
# 回放概率、渲染与 sim2real 开关”等实验变量统一暴露给命令行。
# 上游输入是训练脚本的 parser；下游消费者包括环境构造逻辑、奖励整形、模型编码器和评估/部署工具。

from sample_factory.utils.utils import str2bool


def quadrotors_override_defaults(env, parser):
    # 这里覆盖的是 Sample Factory 的一部分通用默认值，
    # 目的是让框架默认配置更贴近这个项目的观测结构和训练规模，而不是沿用通用 Atari / Mujoco 风格设置。
    #
    # `encoder_type='mlp'` 和 `encoder_subtype='mlp_quads'` 会把默认编码器切到项目自定义的四旋翼观测编码路径；
    # `rnn_size=256` 决定策略网络里循环状态的容量；
    # `encoder_extra_fc_layers=0` 表示不再额外叠加通用全连接层，避免覆盖项目自己在模型文件里定义的编码结构；
    # `env_frameskip=1` 则保证每个策略动作都对应一次物理环境 step，避免额外跳帧破坏动力学时间尺度。
    parser.set_defaults(
        encoder_type='mlp',
        encoder_subtype='mlp_quads',
        rnn_size=256,
        encoder_extra_fc_layers=0,
        env_frameskip=1,
    )


# noinspection PyUnusedLocal
def add_quadrotors_env_args(env, parser):
    # 这里的 `parser` 是训练脚本第一阶段解析后留下来的命令行解析器。
    # 后面添加的每一个参数，最终都会被写进 `cfg`，并沿着环境创建或模型创建路径向下流动。
    p = parser

    # 这一组参数决定“单个 agent 自己看到什么”和“一次 episode 持续多久”。
    # 它们会直接改变 observation 的维度、物理任务时长，以及模型第一层读取的输入结构。
    p.add_argument('--quads_num_agents', default=8, type=int, help='Override default value for the number of quadrotors')
    p.add_argument('--quads_obs_repr', default='xyz_vxyz_R_omega', type=str,
                   choices=['xyz_vxyz_R_omega', 'xyz_vxyz_R_omega_floor', 'xyz_vxyz_R_omega_wall'],
                   help='obs space for quadrotor self')
    p.add_argument('--quads_episode_duration', default=15.0, type=float,
                   help='Override default value for episode duration')
    p.add_argument('--quads_encoder_type', default="corl", type=str, help='The type of the neighborhood encoder')

    # 这一组参数控制邻居信息是否进入观测、每架无人机保留多少邻居槽位，以及邻居信息如何被编码。
    # 它们共同决定论文里 multi-agent 感知链路的宽度：
    # 是完全看不到邻居、看到全部邻居，还是只保留最近的几个邻居给策略网络。
    p.add_argument('--quads_neighbor_visible_num', default=-1, type=int, help='Number of neighbors to consider. -1=all '
                                                                          '0=blind agents, '
                                                                          '0<n<num_agents-1 = nonzero number of agents')
    p.add_argument('--quads_neighbor_obs_type', default='none', type=str,
                   choices=['none', 'pos_vel'], help='Choose what kind of obs to send to encoder.')

    # 这里的 hidden size 和 encoder type 不改变环境状态本身，
    # 但会直接改变 `swarm_rl/models/quad_multi_model.py` 里邻居特征如何压缩、聚合和送入策略头。
    p.add_argument('--quads_neighbor_hidden_size', default=256, type=int,
                   help='The hidden size for the neighbor encoder')
    p.add_argument('--quads_neighbor_encoder_type', default='attention', type=str,
                   choices=['attention', 'mean_embed', 'mlp', 'no_encoder'],
                   help='The type of the neighborhood encoder')

    # 这一组参数决定机间碰撞如何被折算进奖励。
    # `quads_collision_reward` 对应离散碰撞惩罚；
    # `hitbox_radius` 和 `falloff_radius` 把“距离多近算危险”写成可调的物理尺度；
    # `smooth_max_penalty` 则给连续接近惩罚设置上界，避免距离极小时奖励爆掉。
    p.add_argument('--quads_collision_reward', default=0.0, type=float,
                   help='Override default value for quadcol_bin reward, which means collisions between quadrotors')
    p.add_argument('--quads_collision_hitbox_radius', default=2.0, type=float,
                   help='When the distance between two drones are less than N arm_length, we would view them as '
                        'collide.')
    p.add_argument('--quads_collision_falloff_radius', default=-1.0, type=float,
                   help='The falloff radius for the smooth penalty. -1.0: no smooth penalty')
    p.add_argument('--quads_collision_smooth_max_penalty', default=10.0, type=float,
                   help='The upper bound of the collision function given distance among drones')

    # 这一组参数决定是否启用障碍场景，以及障碍观测如何进入策略。
    # 在论文主实验里，这部分会影响障碍物数量、尺寸、分布区域，以及是否提供 octomap/SDF 风格局部障碍信息。
    p.add_argument('--quads_use_obstacles', default=False, type=str2bool, help='Use obstacles or not')
    p.add_argument('--quads_obstacle_obs_type', default='none', type=str,
                   choices=['none', 'octomap'], help='Choose what kind of obs to send to encoder.')
    p.add_argument('--quads_obst_density', default=0.2, type=float, help='Obstacle density in the map')
    p.add_argument('--quads_obst_size', default=1.0, type=float, help='The radius of obstacles')
    p.add_argument('--quads_obst_spawn_area', nargs='+', default=[6.0, 6.0], type=float,
                   help='The spawning area of obstacles')

    # 这些 domain randomization 参数不直接改网络结构，
    # 而是在 reset 时改变障碍密度和尺寸的采样范围，让训练分布更宽，减弱策略对单一地图统计特征的过拟合。
    p.add_argument('--quads_domain_random', default=False, type=str2bool, help='Use domain randomization or not')
    p.add_argument('--quads_obst_density_random', default=False, type=str2bool, help='Enable obstacle density randomization or not')
    p.add_argument('--quads_obst_density_min', default=0.05, type=float,
                   help='The minimum of obstacle density when enabling domain randomization')
    p.add_argument('--quads_obst_density_max', default=0.2, type=float,
                   help='The maximum of obstacle density when enabling domain randomization')
    p.add_argument('--quads_obst_size_random', default=False, type=str2bool, help='Enable obstacle size randomization or not')
    p.add_argument('--quads_obst_size_min', default=0.3, type=float,
                   help='The minimum obstacle size when enabling domain randomization')
    p.add_argument('--quads_obst_size_max', default=0.6, type=float,
                   help='The maximum obstacle size when enabling domain randomization')

    # 这里控制障碍观测进入模型后的编码宽度和编码方式。
    # 环境侧负责产出固定维度障碍观测，而模型侧如何压缩这些观测，则由这两个参数决定。
    p.add_argument('--quads_obst_hidden_size', default=256, type=int, help='The hidden size for the obstacle encoder')
    p.add_argument('--quads_obst_encoder_type', default='mlp', type=str, help='The type of the obstacle encoder')

    # 这项奖励对应“撞到障碍物”这一失败模式，和机间碰撞惩罚分开控制。
    # 这样实验里可以区分：策略是更怕撞队友，还是更怕撞环境障碍。
    p.add_argument('--quads_obst_collision_reward', default=0.0, type=float,
                   help='Override default value for quadcol_bin_obst reward, which means collisions between quadrotor '
                        'and obstacles')

    # 是否启用下洗气流，会改变多机垂直相对位置带来的空气动力学干扰。
    # 这是把环境从“仅几何避碰”推进到“包含气动耦合”的关键开关之一。
    p.add_argument('--quads_use_downwash', default=False, type=str2bool, help='Apply downwash or not')

    # 是否启用 numba 会影响物理仿真和碰撞等数值逻辑的执行路径，主要作用是提升训练吞吐，而不是改变算法含义。
    p.add_argument('--quads_use_numba', default=False, type=str2bool, help='Whether to use numba for jit or not')

    # `quads_mode` 决定每个 episode 的任务分布：静态同目标、动态异目标、追逃、交换目标、编队、障碍场景等。
    # 它是训练样本分布的高层开关，最终会流向场景生成器，改变 reset 时的初始布局和目标设置。
    p.add_argument('--quads_mode', default='static_same_goal', type=str,
                   choices=['static_same_goal', 'static_diff_goal', 'dynamic_same_goal', 'dynamic_diff_goal',
                            'ep_lissajous3D', 'ep_rand_bezier', 'swarm_vs_swarm', 'swap_goals', 'dynamic_formations',
                            'mix', 'o_uniform_same_goal_spawn', 'o_random',
                            'o_dynamic_diff_goal', 'o_dynamic_same_goal', 'o_diagonal', 'o_static_same_goal',
                            'o_static_diff_goal', 'o_swap_goals', 'o_ep_rand_bezier'],
                   help='Choose which scenario to run. ep = evader pursuit')

    # 房间尺寸会进入环境边界碰撞判断，也会影响可用飞行体积、障碍物生成区域和目标布局范围。
    p.add_argument('--quads_room_dims', nargs='+', default=[10., 10., 10.], type=float,
                   help='Length, width, and height dimensions respectively of the quadrotor env')

    # 这项参数控制碰撞回放机制的启用概率。
    # 值为 0 时，每次 episode 都从新随机状态开始；
    # 值大于 0 时，环境会按概率从历史碰撞片段恢复，从而重复训练高风险场景。
    p.add_argument('--replay_buffer_sample_prob', default=0.0, type=float,
                   help='Probability at which we sample from it rather than resetting the env. Set to 0.0 (default) '
                        'to disable the replay. Set to value in (0.0, 1.0] to use replay buffer')

    # 这项参数控制碰撞惩罚是否随训练步数逐步增强。
    # 它的作用是让策略先学会基本飞行和到达目标，再逐渐提高对危险接近和碰撞的约束强度。
    p.add_argument('--anneal_collision_steps', default=0.0, type=float, help='Anneal collision penalties over this '
                                                                             'many steps. Default (0.0) is no '
                                                                             'annealing')

    # 这一组参数只影响可视化和分析，不直接改变物理环境或策略结构。
    # 其中 `visualize_v_value` 主要服务于训练后价值场可视化分析。
    p.add_argument('--quads_view_mode', nargs='+', default=['topdown', 'chase', 'global'],
                   type=str, choices=['topdown', 'chase', 'side', 'global', 'corner0', 'corner1', 'corner2', 'corner3', 'topdownfollow'],
                   help='Choose which kind of view/camera to use')
    p.add_argument('--quads_render', default=False, type=bool, help='Use render or not')
    p.add_argument('--visualize_v_value', action='store_true', help="Visualize v value map")

    # 这个开关把后续流程导向 sim2real 分支。
    # 开启后，部分模型结构或导出逻辑会朝部署约束靠拢，而不是单纯追求训练期表达能力。
    p.add_argument('--quads_sim2real', default=False, type=str2bool, help='Whether to use sim2real or not')
