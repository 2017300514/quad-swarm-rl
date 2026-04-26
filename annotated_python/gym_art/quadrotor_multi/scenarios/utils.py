import sys

import numpy as np
from numpy import cos, sin
from numba import njit

# 这个文件收拢了场景层最基础的任务枚举和几何辅助函数。
# `scenarios/base.py` 与各个具体场景子类会用这些工具，把抽象的 mode 名称变成可落地的编队类型、尺寸范围和目标坐标。

QUADS_MODE_LIST = ['static_same_goal', 'static_diff_goal',  # static formations
                   'ep_lissajous3D', 'ep_rand_bezier',  # evader pursuit
                   'dynamic_same_goal', 'dynamic_diff_goal', 'dynamic_formations', 'swap_goals',  # dynamic formations
                   'swarm_vs_swarm']  # only support >=2 drones

QUADS_MODE_LIST_SINGLE = ['static_same_goal', 'static_diff_goal',  # static formations
                          'ep_lissajous3D', 'ep_rand_bezier',  # evader pursuit
                          'dynamic_same_goal',  # dynamic formations
                          ]

QUADS_MODE_LIST_OBSTACLES = ['o_random', 'o_static_same_goal']
QUADS_MODE_LIST_OBSTACLES_TEST = ['o_random', 'o_static_same_goal',
                                  'o_swap_goals',
                                  'o_ep_rand_bezier', 'o_dynamic_same_goal']

QUADS_MODE_LIST_OBSTACLES_SINGLE = ['o_random']

QUADS_FORMATION_LIST = ['circle_horizontal', 'circle_vertical_xz', 'circle_vertical_yz', 'sphere', 'grid_horizontal',
                        'grid_vertical_xz', 'grid_vertical_yz', 'cube']

QUADS_FORMATION_LIST_OBSTACLES = ['circle_vertical_xz', 'circle_vertical_yz', 'sphere', 'grid_horizontal',
                                  'grid_vertical_xz', 'grid_vertical_yz', 'cube']

# 这是任务模式到“允许哪些编队、邻机最小/最大间距”之间的中心映射表。
# 上层 reset 时会先查这里，再据此采样 formation 和 formation_size。
quad_arm_size = 0.05
QUADS_PARAMS_DICT = {
    'static_same_goal': [['circle_horizontal'], [0.0, 0.0]],
    'dynamic_same_goal': [['circle_horizontal'], [0.0, 0.0]],
    'ep_lissajous3D': [['circle_horizontal'], [0.0, 0.0]],
    'ep_rand_bezier': [['circle_horizontal'], [0.0, 0.0]],
    'static_diff_goal': [QUADS_FORMATION_LIST, [5 * quad_arm_size, 10 * quad_arm_size]],
    'dynamic_diff_goal': [QUADS_FORMATION_LIST, [5 * quad_arm_size, 10 * quad_arm_size]],
    'swarm_vs_swarm': [QUADS_FORMATION_LIST, [5 * quad_arm_size, 10 * quad_arm_size]],
    'swap_goals': [QUADS_FORMATION_LIST, [8 * quad_arm_size, 16 * quad_arm_size]],
    'dynamic_formations': [QUADS_FORMATION_LIST, [0.0, 20 * quad_arm_size]],
    'run_away': [QUADS_FORMATION_LIST, [5 * quad_arm_size, 10 * quad_arm_size]],

    # Obstacles
    'o_random': [['circle_horizontal'], [0.0, 0.0]],
    'o_static_same_goal': [['circle_horizontal'], [0.0, 0.0]],
    'o_dynamic_same_goal': [['circle_horizontal'], [0.0, 0.0]],
    'o_swap_goals': [QUADS_FORMATION_LIST_OBSTACLES, [8 * quad_arm_size, 16 * quad_arm_size]],
    'o_ep_rand_bezier': [['circle_horizontal'], [0.0, 0.0]],
}


def update_formation_and_max_agent_per_layer(mode):
    # 从当前 mode 允许的编队集合里随机抽一个，并给出该编队的默认分层容量。
    # `num_agents_per_layer` 会影响后面圆形/网格/球形编队如何分层排布。
    formation_index = np.random.randint(low=0, high=len(QUADS_PARAMS_DICT[mode][0]))
    formation = QUADS_FORMATION_LIST[formation_index]
    if formation.startswith("circle"):
        num_agents_per_layer = 8
    elif formation.startswith("grid"):
        num_agents_per_layer = 50
    else:
        # for 3D formations. Specific formations override this
        num_agents_per_layer = 8

    return formation, num_agents_per_layer


def update_layer_dist(low, high):
    # 多层编队的层间距离与平面内编队尺度共用同一数量级约束。
    layer_dist = np.random.uniform(low=low, high=high)
    return layer_dist


@njit
def spherical_coordinate(x, y):
    return [cos(x) * cos(y), sin(x) * cos(y), sin(y)]


@njit
def generate_points(n=3):
    # 为球面编队生成近似均匀的单位球点集，避免简单经纬网带来的极区聚集。
    if n < 3:
        # print("The number of goals can not smaller than 3, The system has cast it to 3")
        n = 3

    x = 0.1 + 1.2 * n

    pts = np.zeros((n, 3))
    start = (-1. + 1. / (n - 1.))
    increment = (2. - 2. / (n - 1.)) / (n - 1.)
    pi = np.pi
    for j in range(n):
        s = start + j * increment
        pts[j] = spherical_coordinate(
            x=s * x, y=pi / 2. * np.sign(s) * (1. - np.sqrt(1. - abs(s)))
        )
    return pts


@njit
def get_sphere_radius(num, dist):
    # 给定期望的最近邻间距，反推出球面编队半径。
    # 这里的经验拟合参数把“点数变化”折算成需要多大球半径才不至于太拥挤。
    A = 1.75388487222762
    B = 0.860487305801679
    C = 10.3632729642351
    D = 0.0920858134405214
    ratio = (A - D) / (1 + (num / C) ** B) + D
    radius = dist / ratio
    return radius


@njit
def get_circle_radius(num, dist):
    # 圆环上相邻两点弦长为 dist 时，对应的圆半径。
    theta = 2 * np.pi / num
    radius = (0.5 * dist) / np.sin(theta / 2)
    return radius


@njit
def get_grid_dim_number(num):
    # 给定一个 agent 数，尽量拆成接近正方形的二维网格尺寸。
    sqrt_goal_num = np.sqrt(num)
    grid_number = int(np.floor(sqrt_goal_num))
    dim_1 = grid_number
    while dim_1 > 1:
        if num % dim_1 == 0:
            break
        else:
            dim_1 -= 1

    dim_2 = num // dim_1
    return dim_1, dim_2


def get_formation_range(mode, formation, num_agents, low, high, num_agents_per_layer):
    # 把“任务配置里给出的邻机间距范围”转换成具体编队形状下的 formation_size 取值范围。
    # 不同几何体的 size 语义不同，所以这里做统一桥接。
    if mode == 'swarm_vs_swarm':
        n = num_agents // 2
    else:
        n = num_agents

    if formation.startswith("circle"):
        formation_size_low = get_circle_radius(num_agents_per_layer, low)
        formation_size_high = get_circle_radius(num_agents_per_layer, high)
    elif formation.startswith("grid"):
        formation_size_low = low
        formation_size_high = high
    elif formation.startswith("sphere"):
        formation_size_low = get_sphere_radius(n, low)
        formation_size_high = get_sphere_radius(n, high)
    elif formation.startswith("cube"):
        formation_size_low = low
        formation_size_high = high
    else:
        raise NotImplementedError(f'{formation} is not supported!')

    return formation_size_low, formation_size_high


def get_goal_by_formation(formation, pos_0, pos_1, layer_pos=0.):
    # 把二维编队坐标嵌入到具体朝向平面里：
    # horizontal / vertical_xz / vertical_yz 的区别，本质上就是哪个轴承担层间位移。
    if formation.endswith("horizontal"):
        goal = np.array([pos_0, pos_1, layer_pos])
    elif formation.endswith("vertical_xz"):
        goal = np.array([pos_0, layer_pos, pos_1])
    elif formation.endswith("vertical_yz"):
        goal = np.array([layer_pos, pos_0, pos_1])
    else:
        raise NotImplementedError("Unknown formation")

    return goal


def get_z_value(num_agents, num_agents_per_layer, box_size, formation, formation_size):
    # 给场景中心的 z 轴取值加一个与编队体积相关的下界，避免大编队一生成就穿地板或贴天花板。
    z = np.random.uniform(low=-0.5 * box_size, high=0.5 * box_size) + 2.0
    z_lower_bound = 0.25
    if formation == "sphere" or formation.startswith("circle_vertical"):
        z_lower_bound = formation_size + 0.25
    elif formation.startswith("grid_vertical"):
        real_num_per_layer = np.minimum(num_agents, num_agents_per_layer)
        dim_1, _ = get_grid_dim_number(real_num_per_layer)
        z_lower_bound = dim_1 * formation_size + 0.25

    z = max(z_lower_bound, z)
    return z


def main():
    import timeit
    SETUP_CODE = '''from __main__ import get_circle_radius'''

    TEST_CODE = '''get_circle_radius(num=8, dist=1.1)'''

    # timeit.repeat statement
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=5,
                          number=int(1e5))

    # printing minimum exec. time
    print('times:   ', times)
    print('mean times:   ', np.mean(times[1:]))


if __name__ == '__main__':
    sys.exit(main())
