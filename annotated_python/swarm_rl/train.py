# 中文注释副本；原始文件：swarm_rl/train.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件是整个训练链路的最外层入口。
# 它本身不实现四旋翼动力学、奖励或模型结构，而是负责把“环境注册”“模型注册”“项目专用参数补充”
# 这三件事接到 Sample Factory 的标准训练流程上，最终启动 APPO/IPPO 训练。
# 上游输入主要来自命令行参数；下游输出是传给 `run_rl` 的完整配置，以及已经完成注册的环境/模型工厂。

"""
Main script for training a swarm of quadrotors with SampleFactory

"""

import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.env_wrappers.quadrotor_params import add_quadrotors_env_args, quadrotors_override_defaults
from swarm_rl.models.quad_multi_model import register_models


def register_swarm_components():
    # Sample Factory 通过字符串名字查找环境和模型实现。
    # 这里先把项目里的多机四旋翼环境工厂注册成 `quadrotor_multi`，
    # 再把自定义网络结构注册进模型表；后面的配置文件和命令行参数只会引用这些名字，
    # 不会直接导入底层实现。
    register_env("quadrotor_multi", make_quadrotor_env)
    register_models()


def parse_swarm_cfg(argv=None, evaluation=False):
    # 第一阶段解析先走 Sample Factory 自带参数体系，拿到基础 parser 和部分配置。
    # 这里最关键的中间产物不是数值本身，而是 `partial_cfg.env`：
    # 它告诉项目当前在解析哪一种环境，于是后面才能把四旋翼专用参数挂到 parser 上。
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)

    # 这里把四旋翼项目自己的参数面补进 parser。
    # 这些参数会流向环境构造、邻居观测、障碍物观测、碰撞奖励、回放机制和渲染设置。
    add_quadrotors_env_args(partial_cfg.env, parser)

    # Sample Factory 默认是通用 RL 框架配置，这里用四旋翼项目自己的默认值覆盖其中一部分模型与 rollout 设置。
    # 例如编码器类型、RNN 规模和 env_frameskip 都会影响后续策略网络结构与物理步长对应关系。
    quadrotors_override_defaults(partial_cfg.env, parser)

    # 第二阶段解析重新扫描参数，把刚才补充的四旋翼选项一起写入最终配置对象。
    # 从这里返回的 `final_cfg` 才是训练主循环真正消费的配置；后续环境、模型、PPO 超参数都从它读取。
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():
    """Script entry point."""
    # 先完成注册，再启动解析。
    # 如果先解析再注册，配置虽然能成功读出，但训练循环在实例化环境或模型时会找不到对应实现。
    register_swarm_components()

    # 这里得到的是整个训练过程共享的总配置。
    # 它会沿着 `run_rl -> env creation/model creation -> rollout/update` 这条链路向下传递。
    cfg = parse_swarm_cfg(evaluation=False)

    # `run_rl` 接手后，训练正式进入框架主循环：
    # 创建采样器、实例化多机环境、构建策略网络、执行 rollout、计算优势并做 PPO 更新。
    # 这个脚本到这里为止，职责就从“项目接线”切换成“交给框架驱动”。
    status = run_rl(cfg)
    return status


if __name__ == '__main__':
    # 命令行执行时，把训练状态作为进程退出码链路的一部分返回出去，
    # 便于外部脚本或实验调度器判断本次训练是否正常结束。
    sys.exit(main())
