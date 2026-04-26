import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario

# 这是“目标中心基本不动，但编队尺度连续伸缩”的动态场景。
# 相比 `dynamic_diff_goal` 或 `swap_goals` 这类离散切换任务，它不改目标集合的拓扑，
# 而是把难点放在“同一编队在 episode 中持续变大/变小”上，要求策略一边靠近目标，一边实时调整相对间距。


class Scenario_dynamic_formations(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # `increase_formation_size` 控制当前是在扩张还是收缩编队。
        self.increase_formation_size = True
        # 这里采样的是一个无量纲缩放速度，后面会乘到每个 control step 的 0.001 尺度增量上。
        self.control_speed = np.random.uniform(low=1.0, high=3.0)

    def update_goals(self):
        # `formation_center` 不变时，目标变化完全来自 `formation_size`。
        # 因此每次更新都要重新展开整组几何坐标，再立刻同步回各个子环境。
        self.goals = self.generate_goals(self.num_agents, self.formation_center, layer_dist=self.layer_dist)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self):
        # 到达上/下界后翻转方向，并重新采一个速度。
        # 实现上这里的下界判断写成了 `<= -highest_formation_size`，因此真正触发的主要是上界反向；
        # 注释副本保留源码语义，不改实现。
        if self.formation_size <= -self.highest_formation_size:
            self.increase_formation_size = True
            self.control_speed = np.random.uniform(low=1.0, high=3.0)
        elif self.formation_size >= self.highest_formation_size:
            self.increase_formation_size = False
            self.control_speed = np.random.uniform(low=1.0, high=3.0)

        # 每个控制步只推进一个很小的尺度增量，让编队尺寸是连续变化而不是跳变。
        if self.increase_formation_size:
            self.formation_size += 0.001 * self.control_speed
        else:
            self.formation_size -= 0.001 * self.control_speed

        # 这类任务的核心压力来自“每步都要重算目标几何”。
        # 因而上层的 distance-to-goal 轨迹会反映一个不断移动的相对目标，而不是固定终点。
        self.update_goals()
        return

    def reset(self):
        # 每个 episode 随机初始伸缩方向和伸缩速度，让策略不要只适应单一呼吸节奏。
        self.increase_formation_size = True if np.random.uniform(low=0.0, high=1.0) < 0.5 else False
        self.control_speed = np.random.uniform(low=1.0, high=3.0)

        # 其余部分仍沿用场景基类：抽编队、定中心、生成初始 goals。
        self.standard_reset()

    def update_formation_size(self, new_formation_size):
        # 这个入口主要服务于渲染器或外部交互调节。
        # 一旦外部覆盖 `formation_size`，就立即重建全部 goals，避免场景内部状态和子环境目标脱节。
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.update_goals()
