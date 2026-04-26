# 中文注释副本；原始文件：gym_art/quadrotor_multi/scenarios/dynamic_diff_goal.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np

# 导入当前模块依赖。
from gym_art.quadrotor_multi.scenarios.utils import get_z_value
from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


# 定义类 `Scenario_dynamic_diff_goal`。
class Scenario_dynamic_diff_goal(QuadrotorScenario):
    # 定义函数 `__init__`。
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        # 调用 `super` 执行当前处理。
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # teleport every [4.0, 6.0] secs
        # 保存或更新 `duration_time` 的值。
        duration_time = 5.0
        # 保存或更新 `control_step_for_sec` 的值。
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    # 定义函数 `update_goals`。
    def update_goals(self):
        # Reset formation and related parameters
        # 调用 `update_formation_and_relate_param` 执行当前处理。
        self.update_formation_and_relate_param()

        # Reset goals
        # 保存或更新 `goals` 的值。
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=self.layer_dist)
        # 调用 `shuffle` 执行当前处理。
        np.random.shuffle(self.goals)

    # 定义函数 `step`。
    def step(self):
        # 保存或更新 `tick` 的值。
        tick = self.envs[0].tick
        # 根据条件决定是否进入当前分支。
        if tick % self.control_step_for_sec == 0 and tick > 0:
            # 保存或更新 `box_size` 的值。
            box_size = self.envs[0].box
            # 同时更新 `x`, `y` 等变量。
            x, y = np.random.uniform(low=-box_size, high=box_size, size=(2,))

            # Get z value, and make sure all goals will above the ground
            # 保存或更新 `z` 的值。
            z = get_z_value(num_agents=self.num_agents, num_agents_per_layer=self.num_agents_per_layer,
                            box_size=box_size, formation=self.formation, formation_size=self.formation_size)

            # 保存或更新 `formation_center` 的值。
            self.formation_center = np.array([x, y, z])
            # 调用 `update_goals` 执行当前处理。
            self.update_goals()

            # Update goals to envs
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for i, env in enumerate(self.envs):
                # 保存或更新 `env.goal` 的值。
                env.goal = self.goals[i]

        # 返回当前函数的结果。
        return

    # 定义函数 `reset`。
    def reset(self):
        # Update duration time
        # 保存或更新 `duration_time` 的值。
        duration_time = np.random.uniform(low=4.0, high=6.0)
        # 保存或更新 `control_step_for_sec` 的值。
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        # Reset formation, and parameters related to the formation; formation center; goals
        # 调用 `standard_reset` 执行当前处理。
        self.standard_reset()
