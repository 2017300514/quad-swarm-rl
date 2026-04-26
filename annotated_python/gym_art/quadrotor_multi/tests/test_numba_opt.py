# 中文注释副本；原始文件：gym_art/quadrotor_multi/tests/test_numba_opt.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import time
from unittest import TestCase
import numpy.random as nr

# 导入当前模块依赖。
import numpy

# 导入当前模块依赖。
from gym_art.quadrotor_multi.tests.test_multi_env import create_env
from gym_art.quadrotor_multi.numba_utils import OUNoiseNumba
from gym_art.quadrotor_multi.quad_utils import OUNoise
from gym_art.quadrotor_multi.sensor_noise import SensorNoise


# 定义类 `TestOpt`。
class TestOpt(TestCase):
    # 定义函数 `test_optimized_env`。
    def test_optimized_env(self):
        # 保存或更新 `num_agents` 的值。
        num_agents = 4
        # 保存或更新 `env` 的值。
        env = create_env(num_agents, use_numba=True)

        # 调用 `reset` 执行当前处理。
        env.reset()
        # 调用 `sleep` 执行当前处理。
        time.sleep(0.1)

        # 保存或更新 `num_steps` 的值。
        num_steps = 0
        # 在条件成立时持续执行下面的循环体。
        while num_steps < 100:
            # 同时更新 `obs`, `rewards`, `dones`, `infos` 等变量。
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            # 保存或更新 `num_steps` 的值。
            num_steps += 1
            # 调用 `print` 执行当前处理。
            print('Rewards: ', rewards, "\nCollisions: \n", env.collisions_per_episode)

        # 调用 `close` 执行当前处理。
        env.close()

    # 为下面的函数或方法附加装饰器行为。
    @staticmethod
    # 定义函数 `step_env`。
    def step_env(use_numba, steps):
        # 保存或更新 `num_agents` 的值。
        num_agents = 4
        # 保存或更新 `env` 的值。
        env = create_env(num_agents, use_numba=use_numba)
        # 调用 `reset` 执行当前处理。
        env.reset()
        # 保存或更新 `num_steps` 的值。
        num_steps = 0

        # warmup
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i in range(20):
            # 同时更新 `obs`, `rewards`, `dones`, `infos` 等变量。
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            # 保存或更新 `num_steps` 的值。
            num_steps += 1

        # 调用 `print` 执行当前处理。
        print('Measuring time, numba:', use_numba)
        # 保存或更新 `start` 的值。
        start = time.time()
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i in range(steps):
            # 同时更新 `obs`, `rewards`, `dones`, `infos` 等变量。
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            # this env self-resets

        # 保存或更新 `elapsed_sec` 的值。
        elapsed_sec = time.time() - start
        # 保存或更新 `fps` 的值。
        fps = (num_agents * steps) / elapsed_sec
        # 返回当前函数的结果。
        return fps, elapsed_sec

    # 定义函数 `test_performance_difference`。
    def test_performance_difference(self):
        # 保存或更新 `steps` 的值。
        steps = 1000
        # 同时更新 `fps`, `elapsed_sec` 等变量。
        fps, elapsed_sec = self.step_env(use_numba=False, steps=steps)
        # 同时更新 `fps_numba`, `elapsed_sec_numba` 等变量。
        fps_numba, elapsed_sec_numba = self.step_env(use_numba=True, steps=steps)

        # 调用 `print` 执行当前处理。
        print('Regular: ', fps, elapsed_sec)
        # 调用 `print` 执行当前处理。
        print('Numba: ', fps_numba, elapsed_sec_numba)

    # 定义函数 `test_step_and_noise_opt`。
    def test_step_and_noise_opt(self):
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for _ in range(30):
            # 保存或更新 `num_agents` 的值。
            num_agents = 4
            # 保存或更新 `env` 的值。
            env = create_env(num_agents)
            # 调用 `reset` 执行当前处理。
            env.reset()

            # 保存或更新 `dynamics` 的值。
            dynamics = env.envs[0].dynamics

            # 保存或更新 `dt` 的值。
            dt = 0.005
            # 保存或更新 `thrust_noise_ratio` 的值。
            thrust_noise_ratio = 0.05
            # 保存或更新 `thrusts` 的值。
            thrusts = numpy.random.random(4)

            # 导入当前模块依赖。
            import copy
            # 保存或更新 `dynamics_copy` 的值。
            dynamics_copy = copy.deepcopy(dynamics)
            # 保存或更新 `dynamics_copy_numba` 的值。
            dynamics_copy_numba = copy.deepcopy(dynamics)

            # 保存或更新 `dynamics.thrust_noise` 的值。
            dynamics.thrust_noise = OUNoise(4, sigma=0.2 * thrust_noise_ratio, use_seed=True)
            # 保存或更新 `dynamics_copy_numba.thrust_noise` 的值。
            dynamics_copy_numba.thrust_noise = OUNoiseNumba(4, sigma=0.2 * thrust_noise_ratio, use_seed=True)
            # 保存或更新 `thrust_noise` 的值。
            thrust_noise = thrust_noise_copy = dynamics.thrust_noise.noise()
            # 保存或更新 `thrust_noise_numba` 的值。
            thrust_noise_numba = dynamics_copy_numba.thrust_noise.noise()

            # 调用 `step1` 执行当前处理。
            dynamics.step1(thrusts, dt, thrust_noise)
            # 调用 `step1` 执行当前处理。
            dynamics_copy.step1(thrusts, dt, thrust_noise_copy)
            # 调用 `step1_numba` 执行当前处理。
            dynamics_copy_numba.step1_numba(thrusts, dt, thrust_noise_numba)

            # 定义函数 `pos_vel_acc_tor`。
            def pos_vel_acc_tor(d):
                # 返回当前函数的结果。
                return d.pos, d.vel, d.acc, d.torque

            # 定义函数 `rot_omega_accm`。
            def rot_omega_accm(d):
                # 返回当前函数的结果。
                return d.rot, d.omega, d.accelerometer

            # 同时更新 `p1`, `v1`, `a1`, `t1` 等变量。
            p1, v1, a1, t1 = pos_vel_acc_tor(dynamics)
            # 同时更新 `p2`, `v2`, `a2`, `t2` 等变量。
            p2, v2, a2, t2 = pos_vel_acc_tor(dynamics_copy)
            # 同时更新 `p3`, `v3`, `a3`, `t3` 等变量。
            p3, v3, a3, t3 = pos_vel_acc_tor(dynamics_copy_numba)

            # 调用 `assertTrue` 执行当前处理。
            self.assertTrue(numpy.allclose(p1, p2))
            # 调用 `assertTrue` 执行当前处理。
            self.assertTrue(numpy.allclose(v1, v2))
            # 调用 `assertTrue` 执行当前处理。
            self.assertTrue(numpy.allclose(a1, a2))
            # 调用 `assertTrue` 执行当前处理。
            self.assertTrue(numpy.allclose(t1, t2))

            # 调用 `assertTrue` 执行当前处理。
            self.assertTrue(numpy.allclose(p1, p3))
            # 调用 `assertTrue` 执行当前处理。
            self.assertTrue(numpy.allclose(v1, v3))
            # 调用 `assertTrue` 执行当前处理。
            self.assertTrue(numpy.allclose(a1, a3))
            # 调用 `assertTrue` 执行当前处理。
            self.assertTrue(numpy.allclose(t1, t3))

            # the below test is to check if add_noise is returning the same value
            # 同时更新 `r1`, `o1`, `accm1` 等变量。
            r1, o1, accm1 = rot_omega_accm(dynamics)
            # 同时更新 `r2`, `o2`, `accm2` 等变量。
            r2, o2, accm2 = rot_omega_accm(dynamics_copy_numba)

            # 保存或更新 `sense_noise` 的值。
            sense_noise = SensorNoise(bypass=False, use_numba=False)
            # 保存或更新 `sense_noise_numba` 的值。
            sense_noise_numba = SensorNoise(bypass=False, use_numba=True)

            # 同时更新 `new_p1`, `new_v1`, `new_r1`, `new_o1` 等变量。
            new_p1, new_v1, new_r1, new_o1, new_a1 = sense_noise.add_noise(p1, v1, r1, o1, accm1, dt)
            # 同时更新 `new_p2`, `new_v2`, `new_r2`, `new_o2` 等变量。
            new_p2, new_v2, new_r2, new_o2, new_a2 = sense_noise_numba.add_noise_numba(p2, v2, r2, o2, accm2, dt)

            # 调用 `assertTrue` 执行当前处理。
            self.assertTrue(numpy.allclose(new_p1, new_p2))
            # 调用 `assertTrue` 执行当前处理。
            self.assertTrue(numpy.allclose(new_v1, new_v2))
            # 调用 `assertTrue` 执行当前处理。
            self.assertTrue(numpy.allclose(new_a1, new_a2))
            # 调用 `assertTrue` 执行当前处理。
            self.assertTrue(numpy.allclose(new_o1, new_o2))
            # 调用 `assertTrue` 执行当前处理。
            self.assertTrue(numpy.allclose(new_r1, new_r2))
            # 调用 `close` 执行当前处理。
            env.close()
