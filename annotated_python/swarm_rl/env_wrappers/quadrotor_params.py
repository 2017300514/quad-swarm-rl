# 中文注释副本；原始文件：swarm_rl/env_wrappers/quadrotor_params.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
from sample_factory.utils.utils import str2bool


# 定义函数 `quadrotors_override_defaults`。
def quadrotors_override_defaults(env, parser):
    # 调用 `set_defaults` 执行当前处理。
    parser.set_defaults(
        encoder_type='mlp',
        encoder_subtype='mlp_quads',
        rnn_size=256,
        encoder_extra_fc_layers=0,
        env_frameskip=1,
    )


# noinspection PyUnusedLocal
# 定义函数 `add_quadrotors_env_args`。
def add_quadrotors_env_args(env, parser):
    # 保存或更新 `p` 的值。
    p = parser

    # Quadrotor features
    # 保存或更新 `p.add_argument(--quads_num_agents, default` 的值。
    p.add_argument('--quads_num_agents', default=8, type=int, help='Override default value for the number of quadrotors')
    # 保存或更新 `p.add_argument(--quads_obs_repr, default` 的值。
    p.add_argument('--quads_obs_repr', default='xyz_vxyz_R_omega', type=str,
                   choices=['xyz_vxyz_R_omega', 'xyz_vxyz_R_omega_floor', 'xyz_vxyz_R_omega_wall'],
                   help='obs space for quadrotor self')
    # 保存或更新 `p.add_argument(--quads_episode_duration, default` 的值。
    p.add_argument('--quads_episode_duration', default=15.0, type=float,
                   help='Override default value for episode duration')
    # 保存或更新 `p.add_argument(--quads_encoder_type, default` 的值。
    p.add_argument('--quads_encoder_type', default="corl", type=str, help='The type of the neighborhood encoder')

    # Neighbor
    # Neighbor Features
    # 保存或更新 `p.add_argument(--quads_neighbor_visible_num, default` 的值。
    p.add_argument('--quads_neighbor_visible_num', default=-1, type=int, help='Number of neighbors to consider. -1=all '
                                                                          '0=blind agents, '
                                                                          '0<n<num_agents-1 = nonzero number of agents')
    # 保存或更新 `p.add_argument(--quads_neighbor_obs_type, default` 的值。
    p.add_argument('--quads_neighbor_obs_type', default='none', type=str,
                   choices=['none', 'pos_vel'], help='Choose what kind of obs to send to encoder.')

    # # Neighbor Encoder
    # 保存或更新 `p.add_argument(--quads_neighbor_hidden_size, default` 的值。
    p.add_argument('--quads_neighbor_hidden_size', default=256, type=int,
                   help='The hidden size for the neighbor encoder')
    # 保存或更新 `p.add_argument(--quads_neighbor_encoder_type, default` 的值。
    p.add_argument('--quads_neighbor_encoder_type', default='attention', type=str,
                   choices=['attention', 'mean_embed', 'mlp', 'no_encoder'],
                   help='The type of the neighborhood encoder')

    # # Neighbor Collision Reward
    # 保存或更新 `p.add_argument(--quads_collision_reward, default` 的值。
    p.add_argument('--quads_collision_reward', default=0.0, type=float,
                   help='Override default value for quadcol_bin reward, which means collisions between quadrotors')
    # 保存或更新 `p.add_argument(--quads_collision_hitbox_radius, default` 的值。
    p.add_argument('--quads_collision_hitbox_radius', default=2.0, type=float,
                   help='When the distance between two drones are less than N arm_length, we would view them as '
                        'collide.')
    # 保存或更新 `p.add_argument(--quads_collision_falloff_radius, default` 的值。
    p.add_argument('--quads_collision_falloff_radius', default=-1.0, type=float,
                   help='The falloff radius for the smooth penalty. -1.0: no smooth penalty')
    # 保存或更新 `p.add_argument(--quads_collision_smooth_max_penalty, default` 的值。
    p.add_argument('--quads_collision_smooth_max_penalty', default=10.0, type=float,
                   help='The upper bound of the collision function given distance among drones')

    # Obstacle
    # # Obstacle Features
    # 保存或更新 `p.add_argument(--quads_use_obstacles, default` 的值。
    p.add_argument('--quads_use_obstacles', default=False, type=str2bool, help='Use obstacles or not')
    # 保存或更新 `p.add_argument(--quads_obstacle_obs_type, default` 的值。
    p.add_argument('--quads_obstacle_obs_type', default='none', type=str,
                   choices=['none', 'octomap'], help='Choose what kind of obs to send to encoder.')
    # 保存或更新 `p.add_argument(--quads_obst_density, default` 的值。
    p.add_argument('--quads_obst_density', default=0.2, type=float, help='Obstacle density in the map')
    # 保存或更新 `p.add_argument(--quads_obst_size, default` 的值。
    p.add_argument('--quads_obst_size', default=1.0, type=float, help='The radius of obstacles')
    # 保存或更新 `p.add_argument(--quads_obst_spawn_area, nargs` 的值。
    p.add_argument('--quads_obst_spawn_area', nargs='+', default=[6.0, 6.0], type=float,
                   help='The spawning area of obstacles')
    # 保存或更新 `p.add_argument(--quads_domain_random, default` 的值。
    p.add_argument('--quads_domain_random', default=False, type=str2bool, help='Use domain randomization or not')
    # 保存或更新 `p.add_argument(--quads_obst_density_random, default` 的值。
    p.add_argument('--quads_obst_density_random', default=False, type=str2bool, help='Enable obstacle density randomization or not')
    # 保存或更新 `p.add_argument(--quads_obst_density_min, default` 的值。
    p.add_argument('--quads_obst_density_min', default=0.05, type=float,
                   help='The minimum of obstacle density when enabling domain randomization')
    # 保存或更新 `p.add_argument(--quads_obst_density_max, default` 的值。
    p.add_argument('--quads_obst_density_max', default=0.2, type=float,
                   help='The maximum of obstacle density when enabling domain randomization')
    # 保存或更新 `p.add_argument(--quads_obst_size_random, default` 的值。
    p.add_argument('--quads_obst_size_random', default=False, type=str2bool, help='Enable obstacle size randomization or not')
    # 保存或更新 `p.add_argument(--quads_obst_size_min, default` 的值。
    p.add_argument('--quads_obst_size_min', default=0.3, type=float,
                   help='The minimum obstacle size when enabling domain randomization')
    # 保存或更新 `p.add_argument(--quads_obst_size_max, default` 的值。
    p.add_argument('--quads_obst_size_max', default=0.6, type=float,
                   help='The maximum obstacle size when enabling domain randomization')

    # # Obstacle Encoder
    # 保存或更新 `p.add_argument(--quads_obst_hidden_size, default` 的值。
    p.add_argument('--quads_obst_hidden_size', default=256, type=int, help='The hidden size for the obstacle encoder')
    # 保存或更新 `p.add_argument(--quads_obst_encoder_type, default` 的值。
    p.add_argument('--quads_obst_encoder_type', default='mlp', type=str, help='The type of the obstacle encoder')

    # # Obstacle Collision Reward
    # 保存或更新 `p.add_argument(--quads_obst_collision_reward, default` 的值。
    p.add_argument('--quads_obst_collision_reward', default=0.0, type=float,
                   help='Override default value for quadcol_bin_obst reward, which means collisions between quadrotor '
                        'and obstacles')

    # Aerodynamics
    # # Downwash
    # 保存或更新 `p.add_argument(--quads_use_downwash, default` 的值。
    p.add_argument('--quads_use_downwash', default=False, type=str2bool, help='Apply downwash or not')

    # Numba Speed Up
    # 保存或更新 `p.add_argument(--quads_use_numba, default` 的值。
    p.add_argument('--quads_use_numba', default=False, type=str2bool, help='Whether to use numba for jit or not')

    # Scenarios
    # 保存或更新 `p.add_argument(--quads_mode, default` 的值。
    p.add_argument('--quads_mode', default='static_same_goal', type=str,
                   choices=['static_same_goal', 'static_diff_goal', 'dynamic_same_goal', 'dynamic_diff_goal',
                            'ep_lissajous3D', 'ep_rand_bezier', 'swarm_vs_swarm', 'swap_goals', 'dynamic_formations',
                            'mix', 'o_uniform_same_goal_spawn', 'o_random',
                            'o_dynamic_diff_goal', 'o_dynamic_same_goal', 'o_diagonal', 'o_static_same_goal',
                            'o_static_diff_goal', 'o_swap_goals', 'o_ep_rand_bezier'],
                   help='Choose which scenario to run. ep = evader pursuit')

    # Room
    # 保存或更新 `p.add_argument(--quads_room_dims, nargs` 的值。
    p.add_argument('--quads_room_dims', nargs='+', default=[10., 10., 10.], type=float,
                   help='Length, width, and height dimensions respectively of the quadrotor env')

    # Replay Buffer
    # 保存或更新 `p.add_argument(--replay_buffer_sample_prob, default` 的值。
    p.add_argument('--replay_buffer_sample_prob', default=0.0, type=float,
                   help='Probability at which we sample from it rather than resetting the env. Set to 0.0 (default) '
                        'to disable the replay. Set to value in (0.0, 1.0] to use replay buffer')

    # Annealing
    # 保存或更新 `p.add_argument(--anneal_collision_steps, default` 的值。
    p.add_argument('--anneal_collision_steps', default=0.0, type=float, help='Anneal collision penalties over this '
                                                                             'many steps. Default (0.0) is no '
                                                                             'annealing')

    # Rendering
    # 保存或更新 `p.add_argument(--quads_view_mode, nargs` 的值。
    p.add_argument('--quads_view_mode', nargs='+', default=['topdown', 'chase', 'global'],
                   type=str, choices=['topdown', 'chase', 'side', 'global', 'corner0', 'corner1', 'corner2', 'corner3', 'topdownfollow'],
                   help='Choose which kind of view/camera to use')
    # 保存或更新 `p.add_argument(--quads_render, default` 的值。
    p.add_argument('--quads_render', default=False, type=bool, help='Use render or not')
    # 保存或更新 `p.add_argument(--visualize_v_value, action` 的值。
    p.add_argument('--visualize_v_value', action='store_true', help="Visualize v value map")

    # Sim2Real
    # 保存或更新 `p.add_argument(--quads_sim2real, default` 的值。
    p.add_argument('--quads_sim2real', default=False, type=str2bool, help='Whether to use sim2real or not')
