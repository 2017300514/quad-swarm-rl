# 中文注释副本；原始文件：gym_art/quadrotor_multi/scenarios/test/o_test.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np

# 导入当前模块依赖。
from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


# 定义类 `Scenario_o_test`。
class Scenario_o_test(QuadrotorScenario):
    # 定义函数 `__init__`。
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        # 调用 `super` 执行当前处理。
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # 保存或更新 `start_point` 的值。
        self.start_point = np.array([0.0, -3.0, 2.0])
        # 保存或更新 `end_point` 的值。
        self.end_point = np.array([0.0, 3.0, 2.0])
        # 保存或更新 `room_dims` 的值。
        self.room_dims = room_dims
        # 保存或更新 `duration_time` 的值。
        self.duration_time = 0.0
        # 保存或更新 `quads_mode` 的值。
        self.quads_mode = quads_mode

    # 定义函数 `update_formation_size`。
    def update_formation_size(self, new_formation_size):
        # 当前代码块暂时不执行实际逻辑。
        pass

    # 定义函数 `generate_pos`。
    def generate_pos(self):
        # 保存或更新 `half_room_length` 的值。
        half_room_length = self.room_dims[0] / 2
        # 保存或更新 `half_room_width` 的值。
        half_room_width = self.room_dims[1] / 2

        # 保存或更新 `x` 的值。
        x = np.random.uniform(low=-1.0 * half_room_length + 2.0, high=half_room_length - 2.0)
        # 保存或更新 `y` 的值。
        y = np.random.uniform(low=-1.0 * half_room_width + 2.0, high=half_room_width - 2.0)

        # 保存或更新 `z` 的值。
        z = np.random.uniform(low=1.0, high=4.0)

        # 返回当前函数的结果。
        return np.array([x, y, z])

    # 定义函数 `step`。
    def step(self):
        # 保存或更新 `tick` 的值。
        tick = self.envs[0].tick

        # 根据条件决定是否进入当前分支。
        if tick <= int(self.duration_time * self.envs[0].control_freq):
            # 返回当前函数的结果。
            return

        # 保存或更新 `duration_time` 的值。
        self.duration_time += self.envs[0].ep_time + 1
        # 保存或更新 `goals` 的值。
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)

        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i, env in enumerate(self.envs):
            # 保存或更新 `env.goal` 的值。
            env.goal = self.goals[i]

        # 返回当前函数的结果。
        return

    # 定义函数 `reset`。
    def reset(self):
        # 保存或更新 `start_point` 的值。
        self.start_point = np.array([0.0, 3.0, 2.0])
        # 保存或更新 `end_point` 的值。
        self.end_point = np.array([0.0, -3.0, 2.0])
        # 保存或更新 `duration_time` 的值。
        self.duration_time = np.random.uniform(low=2.0, high=4.0)
        # 保存或更新 `standard_reset(formation_center` 的值。
        self.standard_reset(formation_center=self.start_point)
