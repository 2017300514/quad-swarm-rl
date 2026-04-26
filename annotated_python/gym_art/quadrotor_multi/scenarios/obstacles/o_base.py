# 中文注释副本；原始文件：gym_art/quadrotor_multi/scenarios/obstacles/o_base.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np

# 导入当前模块依赖。
from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


# 定义类 `Scenario_o_base`。
class Scenario_o_base(QuadrotorScenario):
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
        # 保存或更新 `duration_step` 的值。
        self.duration_step = 0
        # 保存或更新 `quads_mode` 的值。
        self.quads_mode = quads_mode
        # 保存或更新 `obstacle_map` 的值。
        self.obstacle_map = None
        # 保存或更新 `free_space` 的值。
        self.free_space = []
        # 保存或更新 `approch_goal_metric` 的值。
        self.approch_goal_metric = 1.0

        # 保存或更新 `spawn_points` 的值。
        self.spawn_points = None
        # 保存或更新 `cell_centers` 的值。
        self.cell_centers = None

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
        if tick <= self.duration_step:
            # 返回当前函数的结果。
            return

        # 保存或更新 `duration_step` 的值。
        self.duration_step += int(self.envs[0].ep_time * self.envs[0].control_freq)
        # 保存或更新 `goals` 的值。
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)

        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i, env in enumerate(self.envs):
            # 保存或更新 `env.goal` 的值。
            env.goal = self.goals[i]

        # 返回当前函数的结果。
        return

    # 定义函数 `reset`。
    def reset(self, obst_map, cell_centers):
        # 保存或更新 `start_point` 的值。
        self.start_point = self.generate_pos()
        # 保存或更新 `end_point` 的值。
        self.end_point = self.generate_pos()
        # 保存或更新 `duration_step` 的值。
        self.duration_step = int(np.random.uniform(low=2.0, high=4.0) * self.envs[0].control_freq)
        # 保存或更新 `standard_reset(formation_center` 的值。
        self.standard_reset(formation_center=self.start_point)

    # 定义函数 `generate_pos_obst_map`。
    def generate_pos_obst_map(self, check_surroundings=False):
        # 保存或更新 `idx` 的值。
        idx = np.random.choice(a=len(self.free_space), replace=False)
        # 同时更新 `x`, `y` 等变量。
        x, y = self.free_space[idx][0], self.free_space[idx][1]
        # 根据条件决定是否进入当前分支。
        if check_surroundings:
            # 保存或更新 `surroundings_free` 的值。
            surroundings_free = self.check_surroundings(x, y)
            # 在条件成立时持续执行下面的循环体。
            while not surroundings_free:
                # 保存或更新 `idx` 的值。
                idx = np.random.choice(a=len(self.free_space), replace=False)
                # 同时更新 `x`, `y` 等变量。
                x, y = self.free_space[idx][0], self.free_space[idx][1]
                # 保存或更新 `surroundings_free` 的值。
                surroundings_free = self.check_surroundings(x, y)

        # 保存或更新 `width` 的值。
        width = self.obstacle_map.shape[0]
        # 保存或更新 `index` 的值。
        index = x + (width * y)
        # 同时更新 `pos_x`, `pos_y` 等变量。
        pos_x, pos_y = self.cell_centers[index]
        # 保存或更新 `z_list_start` 的值。
        z_list_start = np.random.uniform(low=0.75, high=3.0)
        # xy_noise = np.random.uniform(low=-0.2, high=0.2, size=2)
        # 返回当前函数的结果。
        return np.array([pos_x, pos_y, z_list_start])

    # 定义函数 `generate_pos_obst_map_2`。
    def generate_pos_obst_map_2(self, num_agents):
        # 保存或更新 `ids` 的值。
        ids = np.random.choice(range(len(self.free_space)), num_agents, replace=False)

        # 保存或更新 `generated_points` 的值。
        generated_points = []
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for idx in ids:
            # 同时更新 `x`, `y` 等变量。
            x, y = self.free_space[idx][0], self.free_space[idx][1]
            # 保存或更新 `width` 的值。
            width = self.obstacle_map.shape[0]
            # 保存或更新 `index` 的值。
            index = x + (width * y)
            # 同时更新 `pos_x`, `pos_y` 等变量。
            pos_x, pos_y = self.cell_centers[index]
            # 保存或更新 `z_list_start` 的值。
            z_list_start = np.random.uniform(low=1.0, high=3.0)
            # 调用 `append` 执行当前处理。
            generated_points.append(np.array([pos_x, pos_y, z_list_start]))

        # 返回当前函数的结果。
        return np.array(generated_points)

    # 定义函数 `check_surroundings`。
    def check_surroundings(self, row, col):
        # 同时更新 `length`, `width` 等变量。
        length, width = self.obstacle_map.shape[0], self.obstacle_map.shape[1]
        # 保存或更新 `obstacle_map` 的值。
        obstacle_map = self.obstacle_map
        # Check if the given position is out of bounds
        # 根据条件决定是否进入当前分支。
        if row < 0 or row >= width or col < 0 or col >= length:
            # 主动抛出异常以中止或提示错误。
            raise ValueError("Invalid position")

        # Check if the surrounding cells are all 0s
        # 同时更新 `check_pos_x`, `check_pos_y` 等变量。
        check_pos_x, check_pos_y = [], []
        # 根据条件决定是否进入当前分支。
        if row > 0:
            # 调用 `append` 执行当前处理。
            check_pos_x.append(row - 1)
            # 调用 `append` 执行当前处理。
            check_pos_y.append(col)
            # 根据条件决定是否进入当前分支。
            if col > 0:
                # 调用 `append` 执行当前处理。
                check_pos_x.append(row - 1)
                # 调用 `append` 执行当前处理。
                check_pos_y.append(col - 1)
            # 根据条件决定是否进入当前分支。
            if col < length - 1:
                # 调用 `append` 执行当前处理。
                check_pos_x.append(row - 1)
                # 调用 `append` 执行当前处理。
                check_pos_y.append(col + 1)
        # 根据条件决定是否进入当前分支。
        if row < width - 1:
            # 调用 `append` 执行当前处理。
            check_pos_x.append(row + 1)
            # 调用 `append` 执行当前处理。
            check_pos_y.append(col)

        # 根据条件决定是否进入当前分支。
        if col > 0:
            # 调用 `append` 执行当前处理。
            check_pos_x.append(row)
            # 调用 `append` 执行当前处理。
            check_pos_y.append(col - 1)
        # 根据条件决定是否进入当前分支。
        if col < length - 1:
            # 调用 `append` 执行当前处理。
            check_pos_x.append(row)
            # 调用 `append` 执行当前处理。
            check_pos_y.append(col + 1)
            # 根据条件决定是否进入当前分支。
            if row > 0:
                # 调用 `append` 执行当前处理。
                check_pos_x.append(row - 1)
                # 调用 `append` 执行当前处理。
                check_pos_y.append(col + 1)
            # 根据条件决定是否进入当前分支。
            if row < length - 1:
                # 调用 `append` 执行当前处理。
                check_pos_x.append(row + 1)
                # 调用 `append` 执行当前处理。
                check_pos_y.append(col + 1)

        # 保存或更新 `check_pos` 的值。
        check_pos = ([check_pos_x, check_pos_y])
        # Get the values of the adjacent cells
        # 保存或更新 `adjacent_cells` 的值。
        adjacent_cells = obstacle_map[tuple(check_pos)]

        # 返回当前函数的结果。
        return np.any(adjacent_cells != 0)

    # 定义函数 `max_square_area_center`。
    def max_square_area_center(self):
        # 下面开始文档字符串说明。
        """
        Finds the maximum square area of 0 in a 2D matrix and returns the coordinates
        of the center element of the largest square area.
        """
        # 同时更新 `n`, `m` 等变量。
        n, m = self.obstacle_map.shape
        # Initialize a 2D numpy array to store the maximum size of square submatrices
        # that end at each element of the matrix.
        # 保存或更新 `dp` 的值。
        dp = np.zeros((n, m), dtype=int)
        # Initialize the first row and first column of the dp array
        # 保存或更新 `dp[0]` 的值。
        dp[0] = self.obstacle_map[0]
        # 保存或更新 `dp[:, 0]` 的值。
        dp[:, 0] = self.obstacle_map[:, 0]
        # Initialize variables to store the maximum square area and its center coordinates
        # 保存或更新 `max_size` 的值。
        max_size = 0
        # 保存或更新 `center_x` 的值。
        center_x = 0
        # 保存或更新 `center_y` 的值。
        center_y = 0
        # Fill the remaining entries of the dp array
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i in range(1, n):
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for j in range(1, m):
                # 根据条件决定是否进入当前分支。
                if self.obstacle_map[i][j] == 0:
                    # 保存或更新 `dp[i][j]` 的值。
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    # 根据条件决定是否进入当前分支。
                    if dp[i][j] > max_size:
                        # 保存或更新 `max_size` 的值。
                        max_size = dp[i][j]
                        # 保存或更新 `center_x` 的值。
                        center_x = i - (max_size - 1) // 2
                        # 保存或更新 `center_y` 的值。
                        center_y = j - (max_size - 1) // 2
        # Return the center coordinates of the largest square area as a tuple
        # 保存或更新 `index` 的值。
        index = center_x + (m * center_y)
        # 同时更新 `pos_x`, `pos_y` 等变量。
        pos_x, pos_y = self.cell_centers[index]
        # 保存或更新 `z_list_start` 的值。
        z_list_start = np.random.uniform(low=1.5, high=3.0)
        # 返回当前函数的结果。
        return np.array([pos_x, pos_y, z_list_start])
