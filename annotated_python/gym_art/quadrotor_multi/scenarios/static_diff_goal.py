from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario

# 这是最简单的“不同目标静态基线”。
# 它和 `static_same_goal` 的差别不在是否移动目标，而在 reset 时就把每架无人机分配到不同的编队槽位，
# 因此该场景常作为后续 `dynamic_diff_goal`、`swap_goals` 这类复杂重分配任务的静态对照组。


class Scenario_static_diff_goal(QuadrotorScenario):
    def step(self):
        # 场景层不再改写 goals；外层环境只基于 reset 时那组固定分配去推进动力学、奖励和成功率统计。
        return
