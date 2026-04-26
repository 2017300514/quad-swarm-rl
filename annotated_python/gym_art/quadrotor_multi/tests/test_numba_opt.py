# 中文注释副本；原始文件：gym_art/quadrotor_multi/tests/test_numba_opt.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件属于多机四旋翼仿真环境的一部分，负责环境状态、物理过程或配套工具中的某一环。
# 它的上游通常来自场景配置、动力学状态或训练动作，下游会流向观测构造、奖励结算、碰撞处理或可视化。

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
import time
from unittest import TestCase
import numpy.random as nr

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
import numpy

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
from gym_art.quadrotor_multi.tests.test_multi_env import create_env
from gym_art.quadrotor_multi.numba_utils import OUNoiseNumba
from gym_art.quadrotor_multi.quad_utils import OUNoise
from gym_art.quadrotor_multi.sensor_noise import SensorNoise


# `TestOpt` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class TestOpt(TestCase):
    # `test_optimized_env` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def test_optimized_env(self):
        # 该值来自实验配置，决定环境一次并行维护多少架无人机；后续会影响观测拼接尺寸、邻居筛选范围和碰撞矩阵规模。
        num_agents = 4
        env = create_env(num_agents, use_numba=True)

        env.reset()
        time.sleep(0.1)

        num_steps = 0
        while num_steps < 100:
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            num_steps += 1
            print('Rewards: ', rewards, "\nCollisions: \n", env.collisions_per_episode)

        env.close()

    # 这里通过装饰器把额外框架语义附着到下面的定义上，真正影响的是后续调用方式或注册行为。
    @staticmethod
    # `step_env` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def step_env(use_numba, steps):
        # 该值来自实验配置，决定环境一次并行维护多少架无人机；后续会影响观测拼接尺寸、邻居筛选范围和碰撞矩阵规模。
        num_agents = 4
        env = create_env(num_agents, use_numba=use_numba)
        env.reset()
        num_steps = 0

        # warmup
        for i in range(20):
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            num_steps += 1

        print('Measuring time, numba:', use_numba)
        start = time.time()
        for i in range(steps):
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            # this env self-resets

        elapsed_sec = time.time() - start
        fps = (num_agents * steps) / elapsed_sec
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return fps, elapsed_sec

    # `test_performance_difference` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def test_performance_difference(self):
        steps = 1000
        fps, elapsed_sec = self.step_env(use_numba=False, steps=steps)
        fps_numba, elapsed_sec_numba = self.step_env(use_numba=True, steps=steps)

        print('Regular: ', fps, elapsed_sec)
        print('Numba: ', fps_numba, elapsed_sec_numba)

    # `test_step_and_noise_opt` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def test_step_and_noise_opt(self):
        for _ in range(30):
            # 该值来自实验配置，决定环境一次并行维护多少架无人机；后续会影响观测拼接尺寸、邻居筛选范围和碰撞矩阵规模。
            num_agents = 4
            env = create_env(num_agents)
            env.reset()

            dynamics = env.envs[0].dynamics

            dt = 0.005
            thrust_noise_ratio = 0.05
            thrusts = numpy.random.random(4)

            # 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
            import copy
            dynamics_copy = copy.deepcopy(dynamics)
            dynamics_copy_numba = copy.deepcopy(dynamics)

            dynamics.thrust_noise = OUNoise(4, sigma=0.2 * thrust_noise_ratio, use_seed=True)
            dynamics_copy_numba.thrust_noise = OUNoiseNumba(4, sigma=0.2 * thrust_noise_ratio, use_seed=True)
            thrust_noise = thrust_noise_copy = dynamics.thrust_noise.noise()
            thrust_noise_numba = dynamics_copy_numba.thrust_noise.noise()

            dynamics.step1(thrusts, dt, thrust_noise)
            dynamics_copy.step1(thrusts, dt, thrust_noise_copy)
            dynamics_copy_numba.step1_numba(thrusts, dt, thrust_noise_numba)

            # `pos_vel_acc_tor` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
            def pos_vel_acc_tor(d):
                # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
                return d.pos, d.vel, d.acc, d.torque

            # `rot_omega_accm` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
            def rot_omega_accm(d):
                # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
                return d.rot, d.omega, d.accelerometer

            p1, v1, a1, t1 = pos_vel_acc_tor(dynamics)
            p2, v2, a2, t2 = pos_vel_acc_tor(dynamics_copy)
            p3, v3, a3, t3 = pos_vel_acc_tor(dynamics_copy_numba)

            self.assertTrue(numpy.allclose(p1, p2))
            self.assertTrue(numpy.allclose(v1, v2))
            self.assertTrue(numpy.allclose(a1, a2))
            self.assertTrue(numpy.allclose(t1, t2))

            self.assertTrue(numpy.allclose(p1, p3))
            self.assertTrue(numpy.allclose(v1, v3))
            self.assertTrue(numpy.allclose(a1, a3))
            self.assertTrue(numpy.allclose(t1, t3))

            # the below test is to check if add_noise is returning the same value
            r1, o1, accm1 = rot_omega_accm(dynamics)
            r2, o2, accm2 = rot_omega_accm(dynamics_copy_numba)

            sense_noise = SensorNoise(bypass=False, use_numba=False)
            sense_noise_numba = SensorNoise(bypass=False, use_numba=True)

            new_p1, new_v1, new_r1, new_o1, new_a1 = sense_noise.add_noise(p1, v1, r1, o1, accm1, dt)
            new_p2, new_v2, new_r2, new_o2, new_a2 = sense_noise_numba.add_noise_numba(p2, v2, r2, o2, accm2, dt)

            self.assertTrue(numpy.allclose(new_p1, new_p2))
            self.assertTrue(numpy.allclose(new_v1, new_v2))
            self.assertTrue(numpy.allclose(new_a1, new_a2))
            self.assertTrue(numpy.allclose(new_o1, new_o2))
            self.assertTrue(numpy.allclose(new_r1, new_r2))
            env.close()
