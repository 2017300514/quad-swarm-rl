import numpy as np
import bezier

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario

# 这个场景不再让目标点静止在某个编队槽位，而是让整组目标沿一条随机二阶 Bezier 曲线平滑移动。
# 与 `dynamic_same_goal` 的跳变式中心切换不同，这里目标在 5 秒时间窗内连续漂移，
# 因而策略看到的是“共享移动参考轨迹”，更接近追踪一个持续运动的逃逸目标。


class Scenario_ep_rand_bezier(QuadrotorScenario):
    def step(self):
        # 每隔固定 5 秒重采一条新轨迹；这条轨迹的起点是当前目标位置，终点和控制点在房间自由空间内随机采样。
        tick = self.envs[0].tick
        control_freq = self.envs[0].control_freq
        num_secs = 5
        control_steps = int(num_secs * control_freq)
        t = tick % control_steps

        # `room_dims - formation_size` 给轨迹采样留出编队半径余量，避免目标中心合法但整组编队越界。
        room_dims = np.array(self.room_dims) - self.formation_size
        # 新目标不能离当前目标太近，否则轨迹几乎不动；也不能太远，否则移动速度会超过无人机能稳定跟踪的范围。
        max_dist = min(30, max(room_dims))
        min_dist = max_dist / 2
        if tick % control_steps == 0 or tick == 1:
            # 循环采样直到找到一条完全落在房间内部的 Bezier 控制点组合。
            new_goal_found = False
            while not new_goal_found:
                low, high = np.array([-room_dims[0] / 2, -room_dims[1] / 2, 0]), np.array(
                    [room_dims[0] / 2, room_dims[1] / 2, room_dims[2]])

                # 二阶 Bezier 需要两个未来控制点；源码直接在三维空间采样两个方向向量。
                new_pos = np.random.uniform(low=-high, high=high, size=(2, 3)).reshape(3, 2)

                # 把随机方向归一化后再乘随机长度，相当于给轨迹段加入速度尺度扰动。
                new_pos = new_pos * np.random.randint(min_dist, max_dist + 1) / np.linalg.norm(new_pos, axis=0)
                new_pos = self.goals[0].reshape(3, 1) + new_pos
                lower_bound = np.expand_dims(low, axis=1)
                upper_bound = np.expand_dims(high, axis=1)
                new_goal_found = (new_pos > lower_bound + 0.5).all() and (
                        new_pos < upper_bound - 0.5).all()

            # 轨迹节点由“当前位置 + 两个未来控制点”组成，之后预采样成整个时间窗内的离散目标序列。
            nodes = np.concatenate((self.goals[0].reshape(3, 1), new_pos), axis=1)
            nodes = np.asfortranarray(nodes)
            pts = np.linspace(0, 1, control_steps)
            curve = bezier.Curve(nodes, degree=2)
            self.interp = curve.evaluate_multi(pts)

        if tick % control_steps != 0 and tick > 1:
            # 当前实现让所有 agent 共用同一个瞬时目标点。
            # 因此它更像“全队追同一个移动参照物”，而不是每架机各追一条独立曲线。
            self.goals = np.array([self.interp[:, t] for _ in range(self.num_agents)])

            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

        return

    def update_formation_size(self, new_formation_size):
        # 该场景没有实现运行中重设编队尺寸的逻辑；
        # 轨迹采样时只读取当前 `formation_size` 作为边界余量。
        pass
