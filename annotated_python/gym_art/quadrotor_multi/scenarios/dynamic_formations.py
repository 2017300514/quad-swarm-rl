# 中文注释副本；原始文件：gym_art/quadrotor_multi/scenarios/dynamic_formations.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np

# 导入当前模块依赖。
from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


# 定义类 `Scenario_dynamic_formations`。
class Scenario_dynamic_formations(QuadrotorScenario):
    # 定义函数 `__init__`。
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        # 调用 `super` 执行当前处理。
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # if increase_formation_size is True, increase the formation size
        # else, decrease the formation size
        # 保存或更新 `increase_formation_size` 的值。
        self.increase_formation_size = True
        # low: 0.1m/s, high: 0.3m/s
        # 保存或更新 `control_speed` 的值。
        self.control_speed = np.random.uniform(low=1.0, high=3.0)

    # change formation sizes on the fly
    # 定义函数 `update_goals`。
    def update_goals(self):
        # 保存或更新 `goals` 的值。
        self.goals = self.generate_goals(self.num_agents, self.formation_center, layer_dist=self.layer_dist)
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for env, goal in zip(self.envs, self.goals):
            # 保存或更新 `env.goal` 的值。
            env.goal = goal

    # 定义函数 `step`。
    def step(self):
        # 根据条件决定是否进入当前分支。
        if self.formation_size <= -self.highest_formation_size:
            # 保存或更新 `increase_formation_size` 的值。
            self.increase_formation_size = True
            # 保存或更新 `control_speed` 的值。
            self.control_speed = np.random.uniform(low=1.0, high=3.0)
        # 当上一分支不满足时，继续判断新的条件。
        elif self.formation_size >= self.highest_formation_size:
            # 保存或更新 `increase_formation_size` 的值。
            self.increase_formation_size = False
            # 保存或更新 `control_speed` 的值。
            self.control_speed = np.random.uniform(low=1.0, high=3.0)

        # 根据条件决定是否进入当前分支。
        if self.increase_formation_size:
            # 保存或更新 `formation_size` 的值。
            self.formation_size += 0.001 * self.control_speed
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 保存或更新 `formation_size` 的值。
            self.formation_size -= 0.001 * self.control_speed

        # 调用 `update_goals` 执行当前处理。
        self.update_goals()
        # 返回当前函数的结果。
        return

    # 定义函数 `reset`。
    def reset(self):
        # 保存或更新 `increase_formation_size` 的值。
        self.increase_formation_size = True if np.random.uniform(low=0.0, high=1.0) < 0.5 else False
        # 保存或更新 `control_speed` 的值。
        self.control_speed = np.random.uniform(low=1.0, high=3.0)

        # Reset formation, and parameters related to the formation; formation center; goals
        # 调用 `standard_reset` 执行当前处理。
        self.standard_reset()

    # 定义函数 `update_formation_size`。
    def update_formation_size(self, new_formation_size):
        # 根据条件决定是否进入当前分支。
        if new_formation_size != self.formation_size:
            # 保存或更新 `formation_size` 的值。
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            # 调用 `update_goals` 执行当前处理。
            self.update_goals()
