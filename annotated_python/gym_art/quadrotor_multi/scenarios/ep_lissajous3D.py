import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario

# 这个场景用解析式 Lissajous 曲线驱动目标运动。
# 和 `ep_rand_bezier` 的“每隔一段时间随机采一条新曲线”不同，这里整局都沿一个固定频率组合的周期轨迹走，
# 因而更适合测试策略对可重复三维周期运动的持续跟踪能力。


class Scenario_ep_lissajous3D(QuadrotorScenario):
    @staticmethod
    def lissajous3D(tick, a=0.03, b=0.01, c=0.01, n=2, m=2, phi=90, psi=90):
        # 这个静态工具只负责把“时间 -> 三维偏移量”映射成一条 Lissajous 轨迹。
        # 振幅参数 a/b/c 控制三个轴的摆动范围，n/m 和相位决定轨迹形状与重复周期。
        x = a * np.sin(tick)
        y = b * np.sin(n * tick + phi)
        z = c * np.cos(m * tick + psi)
        return x, y, z

    def step(self):
        # 把物理 tick 换成秒级时间，再喂给解析轨迹函数。
        control_freq = self.envs[0].control_freq
        tick = self.envs[0].tick / control_freq
        x, y, z = self.lissajous3D(tick)

        # 轨迹偏移量叠加在初始 goal 上，因此 reset 中选的 `formation_center` 会决定整条曲线在房间里的绝对位置。
        goal_x, goal_y, goal_z = self.goals[0]
        x_new, y_new, z_new = x + goal_x, y + goal_y, z + goal_z

        # 与 `ep_rand_bezier` 一样，当前实现让所有 agent 共享同一个瞬时参考点。
        self.goals = np.array([[x_new, y_new, z_new] for _ in range(self.num_agents)])

        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return

    def update_formation_size(self, new_formation_size):
        # 这个周期轨迹场景没有实现运行中改编队尺寸的接口。
        pass

    def reset(self):
        # 仍先通过基类工具抽取 formation 参数，保持与其余场景一致的初始化口径。
        self.update_formation_and_relate_param()

        # 这里显式把中心偏到 x=-2 侧，是为了给后续周期运动留出空间，降低一开局就撞墙的概率。
        self.formation_center = np.array([-2.0, 0.0, 2.0])
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=0.0)
