# 中文注释副本；原始文件：gym_art/quadrotor_multi/scenarios/base.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np

# 导入当前模块依赖。
from gym_art.quadrotor_multi.scenarios.utils import QUADS_PARAMS_DICT, update_formation_and_max_agent_per_layer, \
    # 执行这一行逻辑。
    update_layer_dist, get_formation_range, get_goal_by_formation
# 导入当前模块依赖。
from gym_art.quadrotor_multi.scenarios.utils import generate_points, get_grid_dim_number


# 定义类 `QuadrotorScenario`。
class QuadrotorScenario:
    # 定义函数 `__init__`。
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        # 保存或更新 `quads_mode` 的值。
        self.quads_mode = quads_mode
        # 保存或更新 `envs` 的值。
        self.envs = envs
        # 保存或更新 `num_agents` 的值。
        self.num_agents = num_agents
        # 保存或更新 `room_dims` 的值。
        self.room_dims = room_dims
        # 保存或更新 `goals` 的值。
        self.goals = None

        #  Set formation, num_agents_per_layer, lowest_formation_size, highest_formation_size, formation_size,
        #  layer_dist, formation_center
        #  Note: num_agents_per_layer for scalability, the maximum number of agent per layer
        # 保存或更新 `formation` 的值。
        self.formation = None
        # 保存或更新 `formation_center` 的值。
        self.formation_center = None
        # 同时更新 `lowest_formation_size`, `highest_formation_size` 等变量。
        self.lowest_formation_size, self.highest_formation_size = 1.0, 2.0
        # 保存或更新 `formation_size` 的值。
        self.formation_size = 1.0

        # 保存或更新 `num_agents_per_layer` 的值。
        self.num_agents_per_layer = 8
        # 保存或更新 `layer_dist` 的值。
        self.layer_dist = self.lowest_formation_size

        # Aux variables for scenario: pursuit evasion
        # 保存或更新 `interp` 的值。
        self.interp = None
        # Aux variables used in scenarios with obstacles
        # 保存或更新 `spawn_points` 的值。
        self.spawn_points = None
        # 保存或更新 `approch_goal_metric` 的值。
        self.approch_goal_metric = 0.5

    # 定义函数 `name`。
    def name(self):
        # 下面开始文档字符串说明。
        """
        :return: scenario name
        """
        # 返回当前函数的结果。
        return self.__class__.__name__

    # 定义函数 `generate_goals`。
    def generate_goals(self, num_agents, formation_center=None, layer_dist=0.0):
        # 根据条件决定是否进入当前分支。
        if formation_center is None:
            # 保存或更新 `formation_center` 的值。
            formation_center = np.array([0., 0., 2.])

        # 根据条件决定是否进入当前分支。
        if self.formation.startswith("circle"):
            # 根据条件决定是否进入当前分支。
            if num_agents <= self.num_agents_per_layer:
                # 保存或更新 `real_num_per_layer` 的值。
                real_num_per_layer = [num_agents]
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # 保存或更新 `whole_layer_num` 的值。
                whole_layer_num = num_agents // self.num_agents_per_layer
                # 保存或更新 `real_num_per_layer` 的值。
                real_num_per_layer = [self.num_agents_per_layer for _ in range(whole_layer_num)]
                # 保存或更新 `rest_num` 的值。
                rest_num = num_agents % self.num_agents_per_layer
                # 根据条件决定是否进入当前分支。
                if rest_num > 0:
                    # 调用 `append` 执行当前处理。
                    real_num_per_layer.append(rest_num)

            # 保存或更新 `pi` 的值。
            pi = np.pi
            # 保存或更新 `goals` 的值。
            goals = []
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for i in range(num_agents):
                # 保存或更新 `cur_layer_num_agents` 的值。
                cur_layer_num_agents = real_num_per_layer[i // self.num_agents_per_layer]
                # 保存或更新 `degree` 的值。
                degree = 2 * pi * (i % cur_layer_num_agents) / cur_layer_num_agents
                # 保存或更新 `pos_0` 的值。
                pos_0 = self.formation_size * np.cos(degree)
                # 保存或更新 `pos_1` 的值。
                pos_1 = self.formation_size * np.sin(degree)
                # 保存或更新 `goal` 的值。
                goal = get_goal_by_formation(formation=self.formation, pos_0=pos_0, pos_1=pos_1,
                                             layer_pos=(i // self.num_agents_per_layer) * layer_dist)
                # 调用 `append` 执行当前处理。
                goals.append(goal)

            # 保存或更新 `goals` 的值。
            goals = np.array(goals)
            # 保存或更新 `goals` 的值。
            goals += formation_center

        # 当上一分支不满足时，继续判断新的条件。
        elif self.formation == "sphere":
            # 保存或更新 `goals` 的值。
            goals = self.formation_size * np.array(generate_points(num_agents)) + formation_center

        # 当上一分支不满足时，继续判断新的条件。
        elif self.formation.startswith("grid"):
            # 根据条件决定是否进入当前分支。
            if num_agents <= self.num_agents_per_layer:
                # 同时更新 `dim_1`, `dim_2` 等变量。
                dim_1, dim_2 = get_grid_dim_number(num_agents)
                # 保存或更新 `dim_size_each_layer` 的值。
                dim_size_each_layer = [[dim_1, dim_2]]
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # whole layer
                # 保存或更新 `whole_layer_num` 的值。
                whole_layer_num = num_agents // self.num_agents_per_layer
                # 同时更新 `max_dim_1`, `max_dim_2` 等变量。
                max_dim_1, max_dim_2 = get_grid_dim_number(self.num_agents_per_layer)
                # 保存或更新 `dim_size_each_layer` 的值。
                dim_size_each_layer = [[max_dim_1, max_dim_2] for _ in range(whole_layer_num)]

                # deal with the rest of the drones
                # 保存或更新 `rest_num` 的值。
                rest_num = num_agents % self.num_agents_per_layer
                # 根据条件决定是否进入当前分支。
                if rest_num > 0:
                    # 同时更新 `dim_1`, `dim_2` 等变量。
                    dim_1, dim_2 = get_grid_dim_number(rest_num)
                    # 调用 `append` 执行当前处理。
                    dim_size_each_layer.append([dim_1, dim_2])

            # 保存或更新 `goals` 的值。
            goals = []
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for i in range(num_agents):
                # 同时更新 `dim_1`, `dim_2` 等变量。
                dim_1, dim_2 = dim_size_each_layer[i // self.num_agents_per_layer]
                # 保存或更新 `pos_0` 的值。
                pos_0 = self.formation_size * (i % dim_2)
                # 保存或更新 `pos_1` 的值。
                pos_1 = self.formation_size * (int(i / dim_2) % dim_1)
                # 保存或更新 `goal` 的值。
                goal = get_goal_by_formation(formation=self.formation, pos_0=pos_0, pos_1=pos_1,
                                             layer_pos=(i // self.num_agents_per_layer) * layer_dist)
                # 调用 `append` 执行当前处理。
                goals.append(goal)

            # 保存或更新 `mean_pos` 的值。
            mean_pos = np.mean(goals, axis=0)
            # 保存或更新 `goals` 的值。
            goals = goals - mean_pos + formation_center
        # 当上一分支不满足时，继续判断新的条件。
        elif self.formation.startswith("cube"):
            # 保存或更新 `dim_size` 的值。
            dim_size = np.power(num_agents, 1.0 / 3)
            # 保存或更新 `floor_dim_size` 的值。
            floor_dim_size = int(dim_size)
            # 保存或更新 `goals` 的值。
            goals = []
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for i in range(num_agents):
                # 保存或更新 `pos_0` 的值。
                pos_0 = self.formation_size * (int(i / floor_dim_size) % floor_dim_size)
                # 保存或更新 `pos_1` 的值。
                pos_1 = self.formation_size * (i % floor_dim_size)
                # 保存或更新 `goal` 的值。
                goal = np.array(
                    [formation_center[2] + self.formation_size * (i // np.square(floor_dim_size)), pos_0, pos_1])
                # 调用 `append` 执行当前处理。
                goals.append(goal)

            # 保存或更新 `mean_pos` 的值。
            mean_pos = np.mean(goals, axis=0)
            # 保存或更新 `goals` 的值。
            goals = goals - mean_pos + formation_center
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 主动抛出异常以中止或提示错误。
            raise NotImplementedError("Unknown formation")

        # 返回当前函数的结果。
        return goals

    # 定义函数 `update_formation_size`。
    def update_formation_size(self, new_formation_size):
        # 根据条件决定是否进入当前分支。
        if new_formation_size != self.formation_size:
            # 保存或更新 `formation_size` 的值。
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            # 保存或更新 `goals` 的值。
            self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                             layer_dist=self.layer_dist)
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for i, env in enumerate(self.envs):
                # 保存或更新 `env.goal` 的值。
                env.goal = self.goals[i]

    # 定义函数 `update_formation_and_relate_param`。
    def update_formation_and_relate_param(self):
        # Reset formation, num_agents_per_layer, lowest_formation_size, highest_formation_size, formation_size,
        # layer_dist
        # 同时更新 `formation`, `num_agents_per_layer` 等变量。
        self.formation, self.num_agents_per_layer = update_formation_and_max_agent_per_layer(mode=self.quads_mode)
        # QUADS_PARAMS_DICT:
        # Key: quads_mode; Value: 0. formation, 1: [formation_low_size, formation_high_size], 2: episode_time
        # 同时更新 `lowest_dist`, `highest_dist` 等变量。
        lowest_dist, highest_dist = QUADS_PARAMS_DICT[self.quads_mode][1]
        # 同时更新 `lowest_formation_size`, `highest_formation_size` 等变量。
        self.lowest_formation_size, self.highest_formation_size = \
            # 保存或更新 `get_formation_range(mode` 的值。
            get_formation_range(mode=self.quads_mode, formation=self.formation, num_agents=self.num_agents,
                                low=lowest_dist, high=highest_dist, num_agents_per_layer=self.num_agents_per_layer)

        # 保存或更新 `formation_size` 的值。
        self.formation_size = np.random.uniform(low=self.lowest_formation_size, high=self.highest_formation_size)
        # 保存或更新 `layer_dist` 的值。
        self.layer_dist = update_layer_dist(low=self.lowest_formation_size, high=self.highest_formation_size)

    # 定义函数 `step`。
    def step(self):
        # 主动抛出异常以中止或提示错误。
        raise NotImplementedError("Implemented in a specific scenario")

    # 定义函数 `reset`。
    def reset(self):
        # Reset formation and related parameters
        # 调用 `update_formation_and_relate_param` 执行当前处理。
        self.update_formation_and_relate_param()

        # Reset formation center
        # 保存或更新 `formation_center` 的值。
        self.formation_center = np.array([0.0, 0.0, 2.0])

        # Regenerate goals, we don't have to assign goals to the envs,
        # the reset function in quadrotor_multi.py would do that
        # 保存或更新 `goals` 的值。
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=self.layer_dist)
        # 调用 `shuffle` 执行当前处理。
        np.random.shuffle(self.goals)

    # 定义函数 `standard_reset`。
    def standard_reset(self, formation_center=None):
        # Reset formation and related parameters
        # 调用 `update_formation_and_relate_param` 执行当前处理。
        self.update_formation_and_relate_param()

        # Reset formation center
        # 根据条件决定是否进入当前分支。
        if formation_center is None:
            # 保存或更新 `formation_center` 的值。
            self.formation_center = np.array([0.0, 0.0, 2.0])
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `formation_center` 的值。
            self.formation_center = formation_center

        # Regenerate goals, we don't have to assign goals to the envs,
        # the reset function in quadrotor_multi.py would do that
        # 保存或更新 `goals` 的值。
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=self.layer_dist)
        # 调用 `shuffle` 执行当前处理。
        np.random.shuffle(self.goals)
