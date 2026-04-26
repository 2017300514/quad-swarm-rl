import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario

# 这是一个面向测试/可视化的简化障碍场景。
# 它不依赖障碍地图采样，而是把起点和终点固定在房间两侧，再按时间间隔触发一次整组目标切换，
# 方便快速观察无人机在受限空间里的穿越、对穿和碰撞行为。


class Scenario_o_test(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.room_dims = room_dims
        self.duration_time = 0.0
        self.quads_mode = quads_mode

    def update_formation_size(self, new_formation_size):
        # 该测试场景没有实现运行中改编队尺寸的需求。
        pass

    def generate_pos(self):
        # 保留一个连续空间采样工具，便于调试时快速改成随机起终点版本。
        half_room_length = self.room_dims[0] / 2
        half_room_width = self.room_dims[1] / 2

        x = np.random.uniform(low=-1.0 * half_room_length + 2.0, high=half_room_length - 2.0)
        y = np.random.uniform(low=-1.0 * half_room_width + 2.0, high=half_room_width - 2.0)
        z = np.random.uniform(low=1.0, high=4.0)
        return np.array([x, y, z])

    def step(self):
        tick = self.envs[0].tick

        if tick <= int(self.duration_time * self.envs[0].control_freq):
            return

        # 到时后顺延下一个切换窗口，并把整组目标切到另一侧。
        self.duration_time += self.envs[0].ep_time + 1
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)

        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return

    def reset(self):
        # reset 时固定一组左右对穿的起终点，方便做直观的穿越测试。
        self.start_point = np.array([0.0, 3.0, 2.0])
        self.end_point = np.array([0.0, -3.0, 2.0])
        self.duration_time = np.random.uniform(low=2.0, high=4.0)
        self.standard_reset(formation_center=self.start_point)
