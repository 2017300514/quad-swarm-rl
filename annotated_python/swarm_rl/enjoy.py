# 中文注释副本；原始文件：swarm_rl/enjoy.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import sys

# 导入当前模块依赖。
from sample_factory.enjoy import enjoy

# 导入当前模块依赖。
from swarm_rl.train import parse_swarm_cfg, register_swarm_components


# 定义函数 `main`。
def main():
    # 下面的文档字符串用于说明当前模块或代码块。
    """Script entry point."""
    # 调用 `register_swarm_components` 执行当前处理。
    register_swarm_components()
    # 保存或更新 `cfg` 的值。
    cfg = parse_swarm_cfg(evaluation=True)
    # 保存或更新 `status` 的值。
    status = enjoy(cfg)
    # 返回当前函数的结果。
    return status


# 根据条件决定是否进入当前分支。
if __name__ == '__main__':
    # 调用 `exit` 执行当前处理。
    sys.exit(main())
