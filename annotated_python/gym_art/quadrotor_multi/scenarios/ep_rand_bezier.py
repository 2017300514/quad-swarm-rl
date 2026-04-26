# 中文注释副本；原始文件：gym_art/quadrotor_multi/scenarios/ep_rand_bezier.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np
import bezier

# 导入当前模块依赖。
from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


# 定义类 `Scenario_ep_rand_bezier`。
class Scenario_ep_rand_bezier(QuadrotorScenario):
    # 定义函数 `step`。
    def step(self):
        # randomly sample new goal pos in free space and have the goal move there following a bezier curve
        # 保存或更新 `tick` 的值。
        tick = self.envs[0].tick
        # 保存或更新 `control_freq` 的值。
        control_freq = self.envs[0].control_freq
        # 保存或更新 `num_secs` 的值。
        num_secs = 5
        # 保存或更新 `control_steps` 的值。
        control_steps = int(num_secs * control_freq)
        # 保存或更新 `t` 的值。
        t = tick % control_steps
        # 保存或更新 `room_dims` 的值。
        room_dims = np.array(self.room_dims) - self.formation_size
        # min and max distance the goal can spawn away from its current location. 30 = empirical upper bound on
        # velocity that the drones can handle.
        # 保存或更新 `max_dist` 的值。
        max_dist = min(30, max(room_dims))
        # 保存或更新 `min_dist` 的值。
        min_dist = max_dist / 2
        # 根据条件决定是否进入当前分支。
        if tick % control_steps == 0 or tick == 1:
            # sample a new goal pos that's within the room boundaries and satisfies the distance constraint
            # 保存或更新 `new_goal_found` 的值。
            new_goal_found = False
            # 在条件成立时持续执行下面的循环体。
            while not new_goal_found:
                # 同时更新 `low`, `high` 等变量。
                low, high = np.array([-room_dims[0] / 2, -room_dims[1] / 2, 0]), np.array(
                    [room_dims[0] / 2, room_dims[1] / 2, room_dims[2]])
                # need an intermediate point for a deg=2 curve
                # 保存或更新 `new_pos` 的值。
                new_pos = np.random.uniform(low=-high, high=high, size=(2, 3)).reshape(3, 2)
                # add some velocity randomization = random magnitude * unit direction
                # 保存或更新 `new_pos` 的值。
                new_pos = new_pos * np.random.randint(min_dist, max_dist + 1) / np.linalg.norm(new_pos, axis=0)
                # 保存或更新 `new_pos` 的值。
                new_pos = self.goals[0].reshape(3, 1) + new_pos
                # 保存或更新 `lower_bound` 的值。
                lower_bound = np.expand_dims(low, axis=1)
                # 保存或更新 `upper_bound` 的值。
                upper_bound = np.expand_dims(high, axis=1)
                # 保存或更新 `new_goal_found` 的值。
                new_goal_found = (new_pos > lower_bound + 0.5).all() and (
                        new_pos < upper_bound - 0.5).all()  # check bounds that are slightly smaller than the room dims
            # 保存或更新 `nodes` 的值。
            nodes = np.concatenate((self.goals[0].reshape(3, 1), new_pos), axis=1)
            # 保存或更新 `nodes` 的值。
            nodes = np.asfortranarray(nodes)
            # 保存或更新 `pts` 的值。
            pts = np.linspace(0, 1, control_steps)
            # 保存或更新 `curve` 的值。
            curve = bezier.Curve(nodes, degree=2)
            # 保存或更新 `interp` 的值。
            self.interp = curve.evaluate_multi(pts)
            # self.interp = np.clip(self.interp, a_min=np.array([0,0,0.2]).reshape(3,1), a_max=high.reshape(3,
            # 1)) # want goal clipping to be slightly above the floor
        # 根据条件决定是否进入当前分支。
        if tick % control_steps != 0 and tick > 1:
            # 保存或更新 `goals` 的值。
            self.goals = np.array([self.interp[:, t] for _ in range(self.num_agents)])

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
