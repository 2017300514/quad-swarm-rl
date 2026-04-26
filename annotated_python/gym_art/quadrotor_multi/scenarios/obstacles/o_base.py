import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario

# 这个文件定义障碍物任务族共享的场景基类。
# 与普通场景基类相比，它除了维护 goals 之外，还要额外维护 `spawn_points`、`obstacle_map`、
# `free_space` 和 `cell_centers`，因为外层 `quadrotor_multi.py` 会先生成障碍地图，再把这些信息交给场景层
# 决定“无人机从哪里起飞、要飞向哪里、何时切换下一段目标”。


class Scenario_o_base(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # 这两个点是无障碍 fallback 默认值；具体障碍子类通常会在 reset 时改写为地图采样出的点。
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.room_dims = room_dims

        # `duration_step` 记录下一次允许切换目标的控制步。
        self.duration_step = 0
        self.quads_mode = quads_mode

        # 这些字段由外层障碍地图生成逻辑提供，再被本类和子类用于采样可行起点/终点。
        self.obstacle_map = None
        self.free_space = []
        self.cell_centers = None

        # 障碍任务里常把接近阈值放宽一些，避免在窄通道里成功判据过于苛刻。
        self.approch_goal_metric = 1.0

        # 外层环境 reset 时会优先读取 `spawn_points` 来设置初始机体位置。
        self.spawn_points = None

    def generate_pos(self):
        # 这个接口用于无障碍连续空间采样，主要给基础/测试场景复用。
        half_room_length = self.room_dims[0] / 2
        half_room_width = self.room_dims[1] / 2

        x = np.random.uniform(low=-1.0 * half_room_length + 2.0, high=half_room_length - 2.0)
        y = np.random.uniform(low=-1.0 * half_room_width + 2.0, high=half_room_width - 2.0)
        z = np.random.uniform(low=1.0, high=4.0)
        return np.array([x, y, z])

    def step(self):
        tick = self.envs[0].tick

        # 时间窗未结束时，维持当前目标不变。
        if tick <= self.duration_step:
            return

        # 超过当前时间窗后，把下一次切换时刻顺延一个 episode 时长，并整组切到 `end_point`。
        self.duration_step += int(self.envs[0].ep_time * self.envs[0].control_freq)
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)

        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return

    def reset(self, obst_map, cell_centers):
        # 这个基类版本更接近无障碍 fallback：直接在连续空间里采样起点终点。
        # 真正依赖障碍地图的子类一般会覆盖它。
        self.start_point = self.generate_pos()
        self.end_point = self.generate_pos()
        self.duration_step = int(np.random.uniform(low=2.0, high=4.0) * self.envs[0].control_freq)
        self.standard_reset(formation_center=self.start_point)

    def generate_pos_obst_map(self, check_surroundings=False):
        # 从自由栅格集合中随机挑一个格子，并映射回真实空间坐标。
        idx = np.random.choice(a=len(self.free_space), replace=False)
        x, y = self.free_space[idx][0], self.free_space[idx][1]
        if check_surroundings:
            # 某些任务希望点位附近也满足额外条件，就继续过滤。
            surroundings_free = self.check_surroundings(x, y)
            while not surroundings_free:
                idx = np.random.choice(a=len(self.free_space), replace=False)
                x, y = self.free_space[idx][0], self.free_space[idx][1]
                surroundings_free = self.check_surroundings(x, y)

        width = self.obstacle_map.shape[0]
        index = x + (width * y)
        pos_x, pos_y = self.cell_centers[index]

        # z 轴单独连续采样，表示同一自由格上方仍允许不同飞行高度。
        z_list_start = np.random.uniform(low=0.75, high=3.0)
        return np.array([pos_x, pos_y, z_list_start])

    def generate_pos_obst_map_2(self, num_agents):
        # 一次性为多架无人机采样互不重复的自由格。
        ids = np.random.choice(range(len(self.free_space)), num_agents, replace=False)

        generated_points = []
        for idx in ids:
            x, y = self.free_space[idx][0], self.free_space[idx][1]
            width = self.obstacle_map.shape[0]
            index = x + (width * y)
            pos_x, pos_y = self.cell_centers[index]
            z_list_start = np.random.uniform(low=1.0, high=3.0)
            generated_points.append(np.array([pos_x, pos_y, z_list_start]))

        return np.array(generated_points)

    def check_surroundings(self, row, col):
        # 检查某个自由格周围是否存在非零单元。
        # 从返回语义看，它更像“周围是否有占用”，而不是真正的“周围完全空闲”。
        length, width = self.obstacle_map.shape[0], self.obstacle_map.shape[1]
        obstacle_map = self.obstacle_map
        if row < 0 or row >= width or col < 0 or col >= length:
            raise ValueError("Invalid position")

        check_pos_x, check_pos_y = [], []
        if row > 0:
            check_pos_x.append(row - 1)
            check_pos_y.append(col)
            if col > 0:
                check_pos_x.append(row - 1)
                check_pos_y.append(col - 1)
            if col < length - 1:
                check_pos_x.append(row - 1)
                check_pos_y.append(col + 1)
        if row < width - 1:
            check_pos_x.append(row + 1)
            check_pos_y.append(col)

        if col > 0:
            check_pos_x.append(row)
            check_pos_y.append(col - 1)
        if col < length - 1:
            check_pos_x.append(row)
            check_pos_y.append(col + 1)
            if row > 0:
                check_pos_x.append(row - 1)
                check_pos_y.append(col + 1)
            if row < length - 1:
                check_pos_x.append(row + 1)
                check_pos_y.append(col + 1)

        check_pos = ([check_pos_x, check_pos_y])
        adjacent_cells = obstacle_map[tuple(check_pos)]
        return np.any(adjacent_cells != 0)

    def max_square_area_center(self):
        # 在障碍地图里找最大连续空白正方形区域的中心点。
        # 这类点适合作为“相对开阔”的出生点或目标点。
        n, m = self.obstacle_map.shape
        dp = np.zeros((n, m), dtype=int)
        dp[0] = self.obstacle_map[0]
        dp[:, 0] = self.obstacle_map[:, 0]

        max_size = 0
        center_x = 0
        center_y = 0
        for i in range(1, n):
            for j in range(1, m):
                if self.obstacle_map[i][j] == 0:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    if dp[i][j] > max_size:
                        max_size = dp[i][j]
                        center_x = i - (max_size - 1) // 2
                        center_y = j - (max_size - 1) // 2

        index = center_x + (m * center_y)
        pos_x, pos_y = self.cell_centers[index]
        z_list_start = np.random.uniform(low=1.5, high=3.0)
        return np.array([pos_x, pos_y, z_list_start])
