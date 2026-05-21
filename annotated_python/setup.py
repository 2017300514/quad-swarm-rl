# 中文注释副本；原始文件：setup.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这个文件不参与训练时的在线计算，但它决定了整个仓库被安装成什么样子。
# 上游是仓库根目录里的 README 与依赖清单，下游是 `pip install -e .` 之后得到的
# `swarm_rl` 包、可导入模块以及一整套训练/评估/绘图/渲染依赖边界。
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# 安装时把仓库 README 读成 PyPI 长描述；这不会影响运行逻辑，但会影响包分发时展示的项目说明。
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# 这里集中固定运行依赖版本。
# 这份列表本质上定义了项目假定的数值/渲染/强化学习软件栈：
# - `sample-factory`、`torch` 负责训练与推理主链；
# - `gymnasium`、`pyglet`、`noise`、`bezier` 等支撑环境与渲染；
# - `plotly`、`matplotlib`、`pandas` 支撑论文分析与可视化。
pip_packages = [
    'numpy==1.26.4', 'matplotlib==3.9.2', 'numba==0.60.0', 'pyglet==1.5.23', 'gym==0.26.2', 'gymnasium==0.28.1',
    'transforms3d==0.4.2', 'noise==1.2.2', 'tqdm==4.66.5', 'Cython==3.0.11', 'scipy==1.14.1',
    'sample-factory>=2.1.1', 'plotly==5.24.1', 'attrdict==2.0.1', 'pandas==2.2.3', 'torch==2.5.0',
    'bezier==2023.7.28', 'typeguard==4.3.0', 'osqp==0.6.7.post3'
]

setup(
    # 包名决定 `pip` 安装后的 import 根命名空间；这里实际导出的是整个 `swarm_rl` 训练栈。
    name='swarm_rl',  # Required

    version='1.0.0',  # Required

    description='Quadrotor Gym Envs',  # Optional

    long_description=long_description,  # Optional

    long_description_content_type='text/markdown',  # Optional

    url='https://github.com/Zhehui-Huang',  # Optional

    author='Zhehui Huang',  # Optional

    author_email='zhehuihu@usc.edu',  # Optional

    keywords='Reinforcement Learning for Quadrotors',  # Optional

    # 自动把仓库里可发现的 Python 包都纳入安装结果。
    # 这使 `swarm_rl`、`gym_art` 等目录在 editable install 后都能被训练脚本直接导入。
    packages=find_packages(where='.'),  # Required

    # Python 版本下界是环境复现的重要约束。
    # 之前排查 conda 环境时之所以选 3.11.11，就是要满足这里的 `>=3.11.10` 要求。
    python_requires='>=3.11.10',

    # `install_requires` 把上面的依赖边界真正交给 `pip`。
    # 训练入口 `swarm_rl.train`、评估入口 `swarm_rl.enjoy`、论文脚本与 OpenGL 渲染链
    # 都默认这些包在环境里已经可导入。
    install_requires=pip_packages,
)
