import numpy as np

from gym_art.quadrotor_multi.scenarios.utils import get_z_value
from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario

# 这是“不同目标 + 周期性整组重采”的动态任务。
# 与 `dynamic_same_goal` 相比，这里不只是把整队平移到一个新中心，而是会同时重采编队几何、
# 目标中心和 agent-goal 对应关系，因此外层环境在结算 reward、distance-to-goal 和成功率时，
# 看到的是一次彻底换任务后的新目标集合。


class Scenario_dynamic_diff_goal(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # 目标组每隔几秒整体“传送”一次，具体间隔会在 reset 时再随机化。
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def update_goals(self):
        # 每次切换都重新抽 formation / formation_size / layer_dist。
        # 这意味着策略不能假设下一轮仍沿用上一轮的几何结构。
        self.update_formation_and_relate_param()

        # 先围绕当前 `formation_center` 生成一整组新目标，再打乱分配顺序。
        # 打乱后，第 i 架无人机下一步追的目标既可能换位置，也可能换槽位语义。
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)

    def step(self):
        tick = self.envs[0].tick
        if tick % self.control_step_for_sec == 0 and tick > 0:
            box_size = self.envs[0].box
            x, y = np.random.uniform(low=-box_size, high=box_size, size=(2,))

            # 新中心的 z 轴不能只随机，否则球形或竖直编队可能直接穿地板。
            # 这里仍通过 `get_z_value` 按当前几何规模估一个安全下界。
            z = get_z_value(num_agents=self.num_agents, num_agents_per_layer=self.num_agents_per_layer,
                            box_size=box_size, formation=self.formation, formation_size=self.formation_size)

            self.formation_center = np.array([x, y, z])
            self.update_goals()

            # 场景层只负责改写目标；真正的轨迹追踪、碰撞修正和指标累积仍由 `quadrotor_multi.py` 继续消费。
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

        return

    def reset(self):
        # 每个 episode 随机一个换目标间隔，避免策略只适应单一节奏。
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        # 初始状态仍复用基类标准 reset：
        # 先抽一套不同目标编队，再由外层环境把这些 goals 分配给各个子环境。
        self.standard_reset()
