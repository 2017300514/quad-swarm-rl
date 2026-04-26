from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario

# 这是最简单的同目标静态场景。
# episode 开始后 goals 不再变化，策略只需要学会从不同初始状态收敛到固定编队目标。


class Scenario_static_same_goal(QuadrotorScenario):
    def update_formation_size(self, new_formation_size):
        # 这个场景没有在 episode 中途动态改编队尺寸的需求，因此留空。
        pass

    def step(self):
        # 静态目标场景每步不额外修改 goal，环境只沿用 reset 时生成的目标点。
        return
