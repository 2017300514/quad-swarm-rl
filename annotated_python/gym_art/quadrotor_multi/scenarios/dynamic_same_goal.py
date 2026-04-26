import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario

# 这是“所有无人机共享同一编队形状，但编队中心会周期性平移”的场景。
# 它常作为基础动态目标任务：策略既要维持相对编队，又要整体跟踪一个不断跳变的新目标中心。


class Scenario_dynamic_same_goal(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)

        # 把“每隔几秒换一次目标中心”换算成控制 step 数，后续直接用 tick 判断是否切换。
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def update_formation_size(self, new_formation_size):
        pass

    def step(self):
        tick = self.envs[0].tick
        if tick % self.control_step_for_sec == 0 and tick > 0:
            # 到达切换时刻后，重新在房间内采样一个新的编队中心。
            box_size = self.envs[0].box
            x, y = np.random.uniform(low=-box_size, high=box_size, size=(2,))
            z = np.random.uniform(low=-0.5 * box_size, high=0.5 * box_size) + 2.0
            z = max(0.25, z)
            self.formation_center = np.array([x, y, z])

            # 同目标场景里所有 agent 共享同一组相对排布，只是整体平移到新的中心位置。
            self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                             layer_dist=0.0)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

        return

    def reset(self):
        # 每个 episode 重新采样换目标的时间间隔，让策略不要只适应固定节奏。
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        # 其余 reset 流程复用基类：抽编队、定中心、生成 goals。
        self.standard_reset()
