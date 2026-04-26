# 中文注释副本；原始文件：gym_art/quadrotor_multi/scenarios/obstacles/o_static_same_goal.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np
import copy

# 导入当前模块依赖。
from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


# 定义类 `Scenario_o_static_same_goal`。
class Scenario_o_static_same_goal(Scenario_o_base):
    # 定义函数 `__init__`。
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        # 调用 `super` 执行当前处理。
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # teleport every [4.0, 6.0] secs
        # 保存或更新 `duration_time` 的值。
        duration_time = 5.0
        # 保存或更新 `control_step_for_sec` 的值。
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)
        # 保存或更新 `approch_goal_metric` 的值。
        self.approch_goal_metric = 1.0

    # 定义函数 `step`。
    def step(self):
        # tick = self.envs[0].tick
        #
        # if tick <= int(self.duration_time * self.envs[0].control_freq):
        #     return
        #
        # self.duration_time += self.envs[0].ep_time + 1
        # for i, env in enumerate(self.envs):
        #     env.goal = self.end_point

        # 返回当前函数的结果。
        return

    # 定义函数 `reset`。
    def reset(self, obst_map=None, cell_centers=None):
        # Update duration time
        # 保存或更新 `duration_time` 的值。
        self.duration_time = np.random.uniform(low=4.0, high=6.0)
        # 保存或更新 `control_step_for_sec` 的值。
        self.control_step_for_sec = int(self.duration_time * self.envs[0].control_freq)

        # 保存或更新 `obstacle_map` 的值。
        self.obstacle_map = obst_map
        # 保存或更新 `cell_centers` 的值。
        self.cell_centers = cell_centers
        # 根据条件决定是否进入当前分支。
        if obst_map is None or cell_centers is None:
            # 主动抛出异常以中止或提示错误。
            raise NotImplementedError

        # 执行这一行逻辑。
        obst_map_locs = np.where(self.obstacle_map == 0)
        # 保存或更新 `free_space` 的值。
        self.free_space = list(zip(*obst_map_locs))

        # 保存或更新 `start_point` 的值。
        self.start_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        # 保存或更新 `end_point` 的值。
        self.end_point = self.max_square_area_center()

        # Reset formation and related parameters
        # 调用 `update_formation_and_relate_param` 执行当前处理。
        self.update_formation_and_relate_param()

        # Reassign goals
        # 保存或更新 `spawn_points` 的值。
        self.spawn_points = copy.deepcopy(self.start_point)
        # 保存或更新 `goals` 的值。
        self.goals = np.array([self.end_point for _ in range(self.num_agents)])
