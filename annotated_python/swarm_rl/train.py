# 中文注释副本；原始文件：swarm_rl/train.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 下面开始文档字符串说明。
"""
Main script for training a swarm of quadrotors with SampleFactory

"""

# 导入当前模块依赖。
import sys

# 导入当前模块依赖。
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

# 导入当前模块依赖。
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.env_wrappers.quadrotor_params import add_quadrotors_env_args, quadrotors_override_defaults
from swarm_rl.models.quad_multi_model import register_models

# 定义函数 `register_swarm_components`。
def register_swarm_components():
    
    # 调用 `register_env` 执行当前处理。
    register_env("quadrotor_multi", make_quadrotor_env)
    # 调用 `register_models` 执行当前处理。
    register_models()

# 定义函数 `parse_swarm_cfg`。
def parse_swarm_cfg(argv=None, evaluation=False):
    # 同时更新 `parser`, `partial_cfg` 等变量。
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    # 调用 `add_quadrotors_env_args` 执行当前处理。
    add_quadrotors_env_args(partial_cfg.env, parser)
    # 调用 `quadrotors_override_defaults` 执行当前处理。
    quadrotors_override_defaults(partial_cfg.env, parser)
    # 保存或更新 `final_cfg` 的值。
    final_cfg = parse_full_cfg(parser, argv)
    # 返回当前函数的结果。
    return final_cfg

# 定义函数 `main`。
def main():
    # 下面的文档字符串用于说明当前模块或代码块。
    """Script entry point."""
    # 调用 `register_swarm_components` 执行当前处理。
    register_swarm_components()
    # 保存或更新 `cfg` 的值。
    cfg = parse_swarm_cfg(evaluation=False)
    # 保存或更新 `status` 的值。
    status = run_rl(cfg)
    # 返回当前函数的结果。
    return status


# 根据条件决定是否进入当前分支。
if __name__ == '__main__':
    # 调用 `exit` 执行当前处理。
    sys.exit(main())
