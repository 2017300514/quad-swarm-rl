# 中文注释副本；原始文件：gym_art/quadrotor_multi/scenarios/run_away.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np

# 导入当前模块依赖。
from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


# 定义类 `Scenario_run_away`。
class Scenario_run_away(QuadrotorScenario):
    # 定义函数 `__init__`。
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        # 调用 `super` 执行当前处理。
        super().__init__(quads_mode, envs, num_agents, room_dims)

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
        # 保存或更新 `tick` 的值。
        tick = self.envs[0].tick
        # 保存或更新 `control_step_for_sec` 的值。
        control_step_for_sec = int(1.0 * self.envs[0].control_freq)

        # 根据条件决定是否进入当前分支。
        if tick % control_step_for_sec == 0 and tick > 0:
            # 保存或更新 `g_index` 的值。
            g_index = np.random.randint(low=1, high=self.num_agents, size=2)
            # 保存或更新 `goals[0]` 的值。
            self.goals[0] = self.goals[g_index[0]]
            # 保存或更新 `goals[1]` 的值。
            self.goals[1] = self.goals[g_index[1]]
            # 保存或更新 `envs[0].goal` 的值。
            self.envs[0].goal = self.goals[0]
            # 保存或更新 `envs[1].goal` 的值。
            self.envs[1].goal = self.goals[1]

        # 返回当前函数的结果。
        return

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

    # 定义函数 `update_formation_size`。
    def update_formation_size(self, new_formation_size):
        # 根据条件决定是否进入当前分支。
        if new_formation_size != self.formation_size:
            # 保存或更新 `formation_size` 的值。
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            # 调用 `update_goals` 执行当前处理。
            self.update_goals()
