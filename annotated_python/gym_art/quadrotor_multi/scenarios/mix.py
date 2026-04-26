# 中文注释副本；原始文件：gym_art/quadrotor_multi/scenarios/mix.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np

# 导入当前模块依赖。
from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario
from gym_art.quadrotor_multi.scenarios.utils import QUADS_MODE_LIST_SINGLE, QUADS_MODE_LIST, \
    # 执行这一行逻辑。
    QUADS_MODE_LIST_OBSTACLES, QUADS_MODE_LIST_OBSTACLES_SINGLE

# Neighbor Scenarios
# 导入当前模块依赖。
from gym_art.quadrotor_multi.scenarios.static_same_goal import Scenario_static_same_goal
from gym_art.quadrotor_multi.scenarios.dynamic_diff_goal import Scenario_dynamic_diff_goal
from gym_art.quadrotor_multi.scenarios.dynamic_formations import Scenario_dynamic_formations
from gym_art.quadrotor_multi.scenarios.dynamic_same_goal import Scenario_dynamic_same_goal
from gym_art.quadrotor_multi.scenarios.ep_lissajous3D import Scenario_ep_lissajous3D
from gym_art.quadrotor_multi.scenarios.ep_rand_bezier import Scenario_ep_rand_bezier
from gym_art.quadrotor_multi.scenarios.run_away import Scenario_run_away
from gym_art.quadrotor_multi.scenarios.static_diff_goal import Scenario_static_diff_goal
from gym_art.quadrotor_multi.scenarios.static_same_goal import Scenario_static_same_goal
from gym_art.quadrotor_multi.scenarios.swap_goals import Scenario_swap_goals
from gym_art.quadrotor_multi.scenarios.swarm_vs_swarm import Scenario_swarm_vs_swarm

# Obstacles
# 导入当前模块依赖。
from gym_art.quadrotor_multi.scenarios.obstacles.o_random import Scenario_o_random
from gym_art.quadrotor_multi.scenarios.obstacles.o_static_same_goal import Scenario_o_static_same_goal
from gym_art.quadrotor_multi.scenarios.obstacles.o_dynamic_same_goal import Scenario_o_dynamic_same_goal
from gym_art.quadrotor_multi.scenarios.obstacles.o_swap_goals import Scenario_o_swap_goals
from gym_art.quadrotor_multi.scenarios.obstacles.o_ep_rand_bezier import Scenario_o_ep_rand_bezier

# Test Scenarios
# 导入当前模块依赖。
from gym_art.quadrotor_multi.scenarios.test.o_test import Scenario_o_test


# 定义函数 `create_scenario`。
def create_scenario(quads_mode, envs, num_agents, room_dims):
    # 保存或更新 `cls` 的值。
    cls = eval('Scenario_' + quads_mode)
    # 保存或更新 `scenario` 的值。
    scenario = cls(quads_mode, envs, num_agents, room_dims)
    # 返回当前函数的结果。
    return scenario


# 定义类 `Scenario_mix`。
class Scenario_mix(QuadrotorScenario):
    # 定义函数 `__init__`。
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        # 调用 `super` 执行当前处理。
        super().__init__(quads_mode, envs, num_agents, room_dims)

        # Once change the parameter here, should also update QUADS_PARAMS_DICT to make sure it is same as run a
        # single scenario key: quads_mode value: 0. formation, 1: [formation_low_size, formation_high_size],
        # 2: episode_time
        # 根据条件决定是否进入当前分支。
        if num_agents == 1:
            # 根据条件决定是否进入当前分支。
            if envs[0].use_obstacles:
                # 保存或更新 `quads_mode_list` 的值。
                self.quads_mode_list = QUADS_MODE_LIST_OBSTACLES_SINGLE
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # 保存或更新 `quads_mode_list` 的值。
                self.quads_mode_list = QUADS_MODE_LIST_SINGLE
        # 当上一分支不满足时，继续判断新的条件。
        elif num_agents > 1 and not envs[0].use_obstacles:
            # 保存或更新 `quads_mode_list` 的值。
            self.quads_mode_list = QUADS_MODE_LIST
        # 当上一分支不满足时，继续判断新的条件。
        elif envs[0].use_obstacles:
            # 保存或更新 `quads_mode_list` 的值。
            self.quads_mode_list = QUADS_MODE_LIST_OBSTACLES
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 主动抛出异常以中止或提示错误。
            raise NotImplementedError("Unknown!")

        # 保存或更新 `scenario` 的值。
        self.scenario = None
        # 保存或更新 `approch_goal_metric` 的值。
        self.approch_goal_metric = 0.5

        # 保存或更新 `spawn_points` 的值。
        self.spawn_points = None

    # 定义函数 `name`。
    def name(self):
        # 下面开始文档字符串说明。
        """
        :return: the name of the actual scenario used in this episode
        """
        # 返回当前函数的结果。
        return self.scenario.__class__.__name__

    # 定义函数 `step`。
    def step(self):
        # 调用 `step` 执行当前处理。
        self.scenario.step()

        # We change goals dynamically
        # 保存或更新 `goals` 的值。
        self.goals = self.scenario.goals

        # Rendering
        # 保存或更新 `formation_size` 的值。
        self.formation_size = self.scenario.formation_size
        # 返回当前函数的结果。
        return

    # 定义函数 `reset`。
    def reset(self, obst_map=None, cell_centers=None):
        # 保存或更新 `mode_index` 的值。
        mode_index = np.random.randint(low=0, high=len(self.quads_mode_list))
        # 保存或更新 `mode` 的值。
        mode = self.quads_mode_list[mode_index]

        # Init the scenario
        # 保存或更新 `scenario` 的值。
        self.scenario = create_scenario(quads_mode=mode, envs=self.envs, num_agents=self.num_agents,
                                        room_dims=self.room_dims)

        # 根据条件决定是否进入当前分支。
        if obst_map is not None:
            # 调用 `reset` 执行当前处理。
            self.scenario.reset(obst_map, cell_centers)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 调用 `reset` 执行当前处理。
            self.scenario.reset()

        # 保存或更新 `goals` 的值。
        self.goals = self.scenario.goals
        # 保存或更新 `spawn_points` 的值。
        self.spawn_points = self.scenario.spawn_points
        # 保存或更新 `formation_size` 的值。
        self.formation_size = self.scenario.formation_size
        # 保存或更新 `approch_goal_metric` 的值。
        self.approch_goal_metric = self.scenario.approch_goal_metric
