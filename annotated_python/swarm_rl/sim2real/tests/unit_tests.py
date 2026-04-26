#!/usr/bin/env python
# 中文注释副本；原始文件：swarm_rl/sim2real/tests/unit_tests.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件是 sim2real 导出链路的对拍测试入口。
# 上游输入来自训练好的 torch 模型目录，以及已经导出的 C 模型源码；
# 下游输出是“PyTorch 前向结果与 C 端前向结果是否一致”的断言。
# 这层不参与训练或导出本身，它负责验证导出流程没有把模型语义改坏。

import torch
import subprocess
import numpy as np
import os

from pathlib import Path
from swarm_rl.sim2real.sim2real import load_sf_model


def compare_torch_to_c_model_outputs_single_drone():
    # 单机测试链路：
    # 先加载训练好的 torch 模型，再把导出的 `model.c` 编译成共享库，最后对同一个随机观测比较两边输出。
    # set this to whatever your project path is
    project_root = Path.home().joinpath('quad-swarm-rl')
    os.chdir(str(project_root))
    # SF torch model used to generate the c model. Set this to be the dir where you store the torch model
    # you used to generate the c model
    torch_model_dir = 'swarm_rl/sim2real/torch_models/single'
    model = load_sf_model(Path(torch_model_dir), model_type='single')

    # 这里用随机自观测跑一遍 PyTorch actor 前向，拿到 4 路推力均值作为基准。
    obs = torch.randn((1, 18))
    obs_dict = {'obs': obs}
    torch_model_out = model.action_parameterization(model.actor_encoder(obs_dict))[1].means.detach().numpy()

    # 再把单机导出的 C 模型即时编译成 `.so`，通过 ctypes 调起同一份前向逻辑。
    c_model_dir = Path('swarm_rl/sim2real/c_models/single')
    c_model_path = c_model_dir.joinpath('model.c')
    shared_lib_path = c_model_dir.joinpath('single.so')
    subprocess.run(
        ['g++', '-fPIC', '-shared', '-o', str(shared_lib_path), str(c_model_path)],
        check=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE
    )

    # 这里通过共享库方式调用 C 端 `main`，模拟导出模型在本地评估时的使用方式。
    import ctypes
    from numpy.ctypeslib import ndpointer
    lib = ctypes.cdll.LoadLibrary(str(shared_lib_path))
    func = lib.main
    func.restype = None
    func.argtypes = [
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")
    ]

    indata = obs.flatten().detach().numpy()
    outdata = np.zeros(4).astype(np.float32)
    func(indata, indata.size, outdata)

    # 只要两边推力均值逐元素足够接近，就说明单机导出路径的数值语义是对齐的。
    assert np.allclose(torch_model_out, outdata)


def compare_torch_to_c_model_multi_drone_attention():
    # attention 版测试更细，不只比较最终 4 路推力，
    # 还会逐层比较邻居编码、障碍编码和注意力 token 输出，便于定位是哪一段导出错了。
    project_root = Path.home().joinpath('quad-swarm-rl')
    os.chdir(str(project_root))

    # 先把 attention 导出的 `model.c` 编译成共享库，供后面多次调用。
    c_model_dir = Path('swarm_rl/sim2real/c_models/attention')
    c_model_path = c_model_dir.joinpath('model.c')
    shared_lib_path = c_model_dir.joinpath('multi_attn.so')
    subprocess.run(
        ['g++', '-fPIC', '-shared', '-o', str(shared_lib_path), str(c_model_path)],
        check=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE
    )

    # C 端 `main` 在这里被当作一个测试钩子：
    # 输入 self / neighbor / obstacle 观测，输出中间 embedding 和最终推力，便于做逐层对拍。
    import ctypes
    from numpy.ctypeslib import ndpointer
    lib = ctypes.cdll.LoadLibrary(str(shared_lib_path))
    func = lib.main
    func.restype = None
    func.argtypes = [
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ]

    torch_model_dir = 'swarm_rl/sim2real/torch_models/attention/'
    model = load_sf_model(Path(torch_model_dir), model_type='attention')

    # 随机重复 1000 次，是为了尽量覆盖不同输入区域，减少“只在某一个样本上看起来对齐”的偶然性。
    for _ in range(1000):
        # 先检查邻居编码器输出是否和 PyTorch 完全一致。
        neighbor_obs = torch.randn(36)
        torch_nbr_out = model.actor_encoder.neighbor_embed_layer(neighbor_obs).detach().numpy()
        nbr_indata = neighbor_obs.detach().numpy()
        nbr_outdata = np.zeros(16).astype(np.float32)

        # 再检查障碍编码器输出。
        obstacle_obs = torch.rand(9)
        torch_obstacle_out = model.actor_encoder.obstacle_embed_layer(obstacle_obs).detach().numpy()
        obst_indata = obstacle_obs.detach().numpy()
        obst_outdata = np.zeros(16).astype(np.float32)  # TODO: make this cfg.rnn_size instead of hardcoded

        # 再把邻居/障碍 embedding 拼成两个 token，检查注意力层输出。
        attn_input = torch.from_numpy(np.vstack((torch_nbr_out, torch_obstacle_out)))
        torch_attn_output, _ = model.actor_encoder.attention_layer(attn_input, attn_input, attn_input)
        # torch_attn_output = model.actor_encoder.attention_layer.softmax_out.detach().numpy()
        torch_attn_output = torch_attn_output.detach().numpy()
        token1_out = np.zeros(16).astype(np.float32)
        token2_out = np.zeros(16).astype(np.float32)

        self_obs = torch.randn(19)
        self_indata = self_obs.detach().numpy()
        obs_dict = {'obs': torch.concat([self_obs, neighbor_obs, obstacle_obs]).view(1, -1)}
        torch_thrust_out = model.action_parameterization(model.actor_encoder(obs_dict))[1].means.flatten().detach().numpy()
        thrust_out = np.zeros(4).astype(np.float32)

        # 这一次调用会同时返回中间 embedding 和最终 thrust，方便做分阶段逐项断言。
        func(self_indata, nbr_indata, obst_indata, nbr_outdata, obst_outdata, token1_out, token2_out, thrust_out)

        tokens = np.vstack((token1_out, token2_out))
        # 四组断言从浅到深覆盖：
        # obstacle encoder、neighbor encoder、attention token、最终 thrust。
        assert np.allclose(torch_obstacle_out, obst_outdata, atol=1e-6)
        assert np.allclose(torch_nbr_out, nbr_outdata, atol=1e-6)
        assert np.allclose(torch_attn_output, tokens, atol=1e-6)
        assert np.allclose(torch_thrust_out, thrust_out, atol=1e-6)


if __name__ == '__main__':
    # 当前入口默认跑 attention 版本对拍。
    # 单机版函数保留在文件中，便于按需切换。
    compare_torch_to_c_model_multi_drone_attention()
    print('Pass Unit Test!')
