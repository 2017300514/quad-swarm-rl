# 中文注释副本；原始文件：gym_art/quadrotor_multi/scenarios/utils.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import sys

# 导入当前模块依赖。
import numpy as np
from numpy import cos, sin
from numba import njit

# 保存或更新 `QUADS_MODE_LIST` 的值。
QUADS_MODE_LIST = ['static_same_goal', 'static_diff_goal',  # static formations
                   'ep_lissajous3D', 'ep_rand_bezier',  # evader pursuit
                   'dynamic_same_goal', 'dynamic_diff_goal', 'dynamic_formations', 'swap_goals',  # dynamic formations
                   'swarm_vs_swarm']  # only support >=2 drones

# 保存或更新 `QUADS_MODE_LIST_SINGLE` 的值。
QUADS_MODE_LIST_SINGLE = ['static_same_goal', 'static_diff_goal',  # static formations
                          'ep_lissajous3D', 'ep_rand_bezier',  # evader pursuit
                          'dynamic_same_goal',  # dynamic formations
                          ]

# 保存或更新 `QUADS_MODE_LIST_OBSTACLES` 的值。
QUADS_MODE_LIST_OBSTACLES = ['o_random', 'o_static_same_goal']
# 保存或更新 `QUADS_MODE_LIST_OBSTACLES_TEST` 的值。
QUADS_MODE_LIST_OBSTACLES_TEST = ['o_random', 'o_static_same_goal',
                                  'o_swap_goals',
                                  'o_ep_rand_bezier', 'o_dynamic_same_goal']

# 保存或更新 `QUADS_MODE_LIST_OBSTACLES_SINGLE` 的值。
QUADS_MODE_LIST_OBSTACLES_SINGLE = ['o_random']

# 保存或更新 `QUADS_FORMATION_LIST` 的值。
QUADS_FORMATION_LIST = ['circle_horizontal', 'circle_vertical_xz', 'circle_vertical_yz', 'sphere', 'grid_horizontal',
                        'grid_vertical_xz', 'grid_vertical_yz', 'cube']

# 保存或更新 `QUADS_FORMATION_LIST_OBSTACLES` 的值。
QUADS_FORMATION_LIST_OBSTACLES = ['circle_vertical_xz', 'circle_vertical_yz', 'sphere', 'grid_horizontal',
                                  'grid_vertical_xz', 'grid_vertical_yz', 'cube']

# key: quads_mode
# value: 0. formation, 1: [formation_low_size, formation_high_size]
# 保存或更新 `quad_arm_size` 的值。
quad_arm_size = 0.05
# 保存或更新 `QUADS_PARAMS_DICT` 的值。
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


# 定义函数 `update_formation_and_max_agent_per_layer`。
def update_formation_and_max_agent_per_layer(mode):
    # 保存或更新 `formation_index` 的值。
    formation_index = np.random.randint(low=0, high=len(QUADS_PARAMS_DICT[mode][0]))
    # 保存或更新 `formation` 的值。
    formation = QUADS_FORMATION_LIST[formation_index]
    # 根据条件决定是否进入当前分支。
    if formation.startswith("circle"):
        # 保存或更新 `num_agents_per_layer` 的值。
        num_agents_per_layer = 8
    # 当上一分支不满足时，继续判断新的条件。
    elif formation.startswith("grid"):
        # 保存或更新 `num_agents_per_layer` 的值。
        num_agents_per_layer = 50
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # for 3D formations. Specific formations override this
        # 保存或更新 `num_agents_per_layer` 的值。
        num_agents_per_layer = 8

    # 返回当前函数的结果。
    return formation, num_agents_per_layer


# 定义函数 `update_layer_dist`。
def update_layer_dist(low, high):
    # 保存或更新 `layer_dist` 的值。
    layer_dist = np.random.uniform(low=low, high=high)
    # 返回当前函数的结果。
    return layer_dist


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `spherical_coordinate`。
def spherical_coordinate(x, y):
    # 返回当前函数的结果。
    return [cos(x) * cos(y), sin(x) * cos(y), sin(y)]


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `generate_points`。
def generate_points(n=3):
    # 根据条件决定是否进入当前分支。
    if n < 3:
        # print("The number of goals can not smaller than 3, The system has cast it to 3")
        # 保存或更新 `n` 的值。
        n = 3

    # 保存或更新 `x` 的值。
    x = 0.1 + 1.2 * n

    # 保存或更新 `pts` 的值。
    pts = np.zeros((n, 3))
    # 保存或更新 `start` 的值。
    start = (-1. + 1. / (n - 1.))
    # 保存或更新 `increment` 的值。
    increment = (2. - 2. / (n - 1.)) / (n - 1.)
    # 保存或更新 `pi` 的值。
    pi = np.pi
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for j in range(n):
        # 保存或更新 `s` 的值。
        s = start + j * increment
        # 保存或更新 `pts[j]` 的值。
        pts[j] = spherical_coordinate(
            x=s * x, y=pi / 2. * np.sign(s) * (1. - np.sqrt(1. - abs(s)))
        )
    # 返回当前函数的结果。
    return pts


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `get_sphere_radius`。
def get_sphere_radius(num, dist):
    # 保存或更新 `A` 的值。
    A = 1.75388487222762
    # 保存或更新 `B` 的值。
    B = 0.860487305801679
    # 保存或更新 `C` 的值。
    C = 10.3632729642351
    # 保存或更新 `D` 的值。
    D = 0.0920858134405214
    # 保存或更新 `ratio` 的值。
    ratio = (A - D) / (1 + (num / C) ** B) + D
    # 保存或更新 `radius` 的值。
    radius = dist / ratio
    # 返回当前函数的结果。
    return radius


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `get_circle_radius`。
def get_circle_radius(num, dist):
    # 保存或更新 `theta` 的值。
    theta = 2 * np.pi / num
    # 保存或更新 `radius` 的值。
    radius = (0.5 * dist) / np.sin(theta / 2)
    # 返回当前函数的结果。
    return radius


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `get_grid_dim_number`。
def get_grid_dim_number(num):
    # 保存或更新 `sqrt_goal_num` 的值。
    sqrt_goal_num = np.sqrt(num)
    # 保存或更新 `grid_number` 的值。
    grid_number = int(np.floor(sqrt_goal_num))
    # 保存或更新 `dim_1` 的值。
    dim_1 = grid_number
    # 在条件成立时持续执行下面的循环体。
    while dim_1 > 1:
        # 根据条件决定是否进入当前分支。
        if num % dim_1 == 0:
            # 提前结束当前循环。
            break
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `dim_1` 的值。
            dim_1 -= 1

    # 保存或更新 `dim_2` 的值。
    dim_2 = num // dim_1
    # 返回当前函数的结果。
    return dim_1, dim_2


# 定义函数 `get_formation_range`。
def get_formation_range(mode, formation, num_agents, low, high, num_agents_per_layer):
    # Numba just makes it ~ 5 times slower
    # 根据条件决定是否进入当前分支。
    if mode == 'swarm_vs_swarm':
        # 保存或更新 `n` 的值。
        n = num_agents // 2
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 保存或更新 `n` 的值。
        n = num_agents

    # 根据条件决定是否进入当前分支。
    if formation.startswith("circle"):
        # 保存或更新 `formation_size_low` 的值。
        formation_size_low = get_circle_radius(num_agents_per_layer, low)
        # 保存或更新 `formation_size_high` 的值。
        formation_size_high = get_circle_radius(num_agents_per_layer, high)
    # 当上一分支不满足时，继续判断新的条件。
    elif formation.startswith("grid"):
        # 保存或更新 `formation_size_low` 的值。
        formation_size_low = low
        # 保存或更新 `formation_size_high` 的值。
        formation_size_high = high
    # 当上一分支不满足时，继续判断新的条件。
    elif formation.startswith("sphere"):
        # 保存或更新 `formation_size_low` 的值。
        formation_size_low = get_sphere_radius(n, low)
        # 保存或更新 `formation_size_high` 的值。
        formation_size_high = get_sphere_radius(n, high)
    # 当上一分支不满足时，继续判断新的条件。
    elif formation.startswith("cube"):
        # 保存或更新 `formation_size_low` 的值。
        formation_size_low = low
        # 保存或更新 `formation_size_high` 的值。
        formation_size_high = high
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 主动抛出异常以中止或提示错误。
        raise NotImplementedError(f'{formation} is not supported!')

    # 返回当前函数的结果。
    return formation_size_low, formation_size_high


# 定义函数 `get_goal_by_formation`。
def get_goal_by_formation(formation, pos_0, pos_1, layer_pos=0.):
    # Numba just makes it ~ 4 times slower
    # 根据条件决定是否进入当前分支。
    if formation.endswith("horizontal"):
        # 保存或更新 `goal` 的值。
        goal = np.array([pos_0, pos_1, layer_pos])
    # 当上一分支不满足时，继续判断新的条件。
    elif formation.endswith("vertical_xz"):
        # 保存或更新 `goal` 的值。
        goal = np.array([pos_0, layer_pos, pos_1])
    # 当上一分支不满足时，继续判断新的条件。
    elif formation.endswith("vertical_yz"):
        # 保存或更新 `goal` 的值。
        goal = np.array([layer_pos, pos_0, pos_1])
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 主动抛出异常以中止或提示错误。
        raise NotImplementedError("Unknown formation")

    # 返回当前函数的结果。
    return goal


# 定义函数 `get_z_value`。
def get_z_value(num_agents, num_agents_per_layer, box_size, formation, formation_size):
    # 保存或更新 `z` 的值。
    z = np.random.uniform(low=-0.5 * box_size, high=0.5 * box_size) + 2.0
    # 保存或更新 `z_lower_bound` 的值。
    z_lower_bound = 0.25
    # 根据条件决定是否进入当前分支。
    if formation == "sphere" or formation.startswith("circle_vertical"):
        # 保存或更新 `z_lower_bound` 的值。
        z_lower_bound = formation_size + 0.25
    # 当上一分支不满足时，继续判断新的条件。
    elif formation.startswith("grid_vertical"):
        # 保存或更新 `real_num_per_layer` 的值。
        real_num_per_layer = np.minimum(num_agents, num_agents_per_layer)
        # 同时更新 `dim_1`, `_` 等变量。
        dim_1, _ = get_grid_dim_number(real_num_per_layer)
        # 保存或更新 `z_lower_bound` 的值。
        z_lower_bound = dim_1 * formation_size + 0.25

    # 保存或更新 `z` 的值。
    z = max(z_lower_bound, z)
    # 返回当前函数的结果。
    return z


# 定义函数 `main`。
def main():
    # 导入当前模块依赖。
    import timeit
    # 保存或更新 `SETUP_CODE` 的值。
    SETUP_CODE = '''from __main__ import get_circle_radius'''

    # 保存或更新 `TEST_CODE` 的值。
    TEST_CODE = '''get_circle_radius(num=8, dist=1.1)'''

    # timeit.repeat statement
    # 保存或更新 `times` 的值。
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=5,
                          number=int(1e5))

    # printing minimum exec. time
    # 调用 `print` 执行当前处理。
    print('times:   ', times)
    # 调用 `print` 执行当前处理。
    print('mean times:   ', np.mean(times[1:]))


# 根据条件决定是否进入当前分支。
if __name__ == '__main__':
    # 调用 `exit` 执行当前处理。
    sys.exit(main())
