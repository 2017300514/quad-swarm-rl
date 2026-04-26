# 中文注释副本；原始文件：gym_art/quadrotor_multi/scenarios/ep_lissajous3D.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np

# 导入当前模块依赖。
from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


# 定义类 `Scenario_ep_lissajous3D`。
class Scenario_ep_lissajous3D(QuadrotorScenario):
    # Based on https://mathcurve.com/courbes3d.gb/lissajous3d/lissajous3d.shtml
    # 为下面的函数或方法附加装饰器行为。
    @staticmethod
    # 定义函数 `lissajous3D`。
    def lissajous3D(tick, a=0.03, b=0.01, c=0.01, n=2, m=2, phi=90, psi=90):
        # 保存或更新 `x` 的值。
        x = a * np.sin(tick)
        # 保存或更新 `y` 的值。
        y = b * np.sin(n * tick + phi)
        # 保存或更新 `z` 的值。
        z = c * np.cos(m * tick + psi)
        # 返回当前函数的结果。
        return x, y, z

    # 定义函数 `step`。
    def step(self):
        # 保存或更新 `control_freq` 的值。
        control_freq = self.envs[0].control_freq
        # 保存或更新 `tick` 的值。
        tick = self.envs[0].tick / control_freq
        # 同时更新 `x`, `y`, `z` 等变量。
        x, y, z = self.lissajous3D(tick)
        # 同时更新 `goal_x`, `goal_y`, `goal_z` 等变量。
        goal_x, goal_y, goal_z = self.goals[0]
        # 同时更新 `x_new`, `y_new`, `z_new` 等变量。
        x_new, y_new, z_new = x + goal_x, y + goal_y, z + goal_z
        # 保存或更新 `goals` 的值。
        self.goals = np.array([[x_new, y_new, z_new] for _ in range(self.num_agents)])

        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i, env in enumerate(self.envs):
            # 保存或更新 `env.goal` 的值。
            env.goal = self.goals[i]

        # 返回当前函数的结果。
        return

    # 定义函数 `update_formation_size`。
    def update_formation_size(self, new_formation_size):
        # 当前代码块暂时不执行实际逻辑。
        pass

    # 定义函数 `reset`。
    def reset(self):
        # Reset formation and related parameters
        # 调用 `update_formation_and_relate_param` 执行当前处理。
        self.update_formation_and_relate_param()

        # Generate goals
        # 保存或更新 `formation_center` 的值。
        self.formation_center = np.array([-2.0, 0.0, 2.0])  # prevent drones from crashing into the wall
        # 保存或更新 `goals` 的值。
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=0.0)
