from __future__ import annotations

"""
Deprecated helper.

This script was used for an earlier template-based annotation pass.
It is no longer the accepted workflow for this repository because the
generated comments cannot reliably capture project-specific semantics.
Formal annotation work must now be done file by file through manual reading.
"""

import ast
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "annotated_python"
SUMMARY_PATH = REPO_ROOT / "ANNOTATION_AND_PAPER_GUIDE.md"
PAPER_PATH = REPO_ROOT / "hybrid_auto" / "Huang 等 - Collision Avoidance and Navigation for a Quadrotor.md"
EXCLUDED_DIRS = {".git", "annotated_python", "__pycache__", "tools"}


MODULE_DESCRIPTIONS = [
    ("swarm_rl/train.py", [
        "该文件是 Sample Factory 训练入口，负责把四旋翼环境、模型注册和命令行配置拼成一次完整的 RL 启动流程。",
        "上游输入来自命令行参数和 `swarm_rl.env_wrappers.quadrotor_params`，下游输出是交给 `run_rl` 的最终训练配置与环境注册表。",
    ]),
    ("swarm_rl/enjoy.py", [
        "该文件负责加载训练后的策略并驱动评估/可视化流程，用来检查训练出的多机避障与导航策略在环境中的实际表现。",
        "它复用训练期的环境与模型注册逻辑，但下游不再执行参数更新，而是消费 checkpoint 做 rollout 和渲染。",
    ]),
    ("swarm_rl/models/quad_multi_model.py", [
        "该文件定义论文中的多机策略/价值网络编码器，是 self、neighbor、obstacle 三类观测如何融合成策略特征的核心位置。",
        "上游输入是环境拼接好的观测向量，下游输出是 Sample Factory 策略头和价值头需要的隐表示。",
    ]),
    ("swarm_rl/models/attention_layer.py", [
        "该文件实现邻居/障碍信息融合使用的注意力算子，决定 token 之间如何交换信息以及最终特征如何聚合。",
        "它被训练模型和 sim2real 轻量模型共同复用，因此这里的张量形状和归一化方式会直接影响部署一致性。",
    ]),
    ("swarm_rl/env_wrappers/quad_utils.py", [
        "该文件把四旋翼环境与 Sample Factory 的接口对接起来，负责按配置构造环境、包装 reward shaping 和经验回放等训练辅助逻辑。",
        "上游是命令行配置，下游是可被训练循环直接实例化的环境工厂。",
    ]),
    ("swarm_rl/env_wrappers/reward_shaping.py", [
        "该文件负责在环境原始奖励之外施加训练期的奖励整形与退火逻辑，用来稳定学习早期的探索并逐步过渡到目标奖励。",
        "这里调整的每一项系数最终都会改变 PPO 收到的回报信号，直接影响碰撞规避和到达目标的权衡。",
    ]),
    ("swarm_rl/env_wrappers/quadrotor_params.py", [
        "该文件集中管理四旋翼实验相关的配置参数，是论文复现时训练入口、环境实现和模型结构之间的公共参数面。",
        "这些参数先由命令行解析得到，再流向环境构造、奖励计算、邻居观测和模型编码器等模块。",
    ]),
    ("gym_art/quadrotor_multi/quadrotor_multi.py", [
        "该文件实现多机四旋翼环境主循环，负责把单机动力学、场景生成、邻居关系、障碍物、碰撞和奖励结算整合成一个 step。",
        "上游输入是动作、场景配置和单机状态；下游输出是策略网络消费的观测、奖励、终止标志以及训练日志统计。",
    ]),
    ("gym_art/quadrotor_multi/quadrotor_single.py", [
        "该文件封装单架四旋翼的底层动力学、控制与基础观测，是多机环境里每个 agent 的最小物理单元。",
        "多机环境会从这里收集位置、速度、姿态等原始状态，再向上拼成邻居观测、碰撞检测和奖励项。",
    ]),
    ("gym_art/quadrotor_multi/quad_experience_replay.py", [
        "该文件实现论文中针对碰撞片段的经验回放机制，用来放大稀有但关键的失败样本。",
        "它从环境里接收碰撞前后的状态快照，下游在新 episode 开始时按概率恢复这些片段，逼策略反复学习高风险场景。",
    ]),
    ("gym_art/quadrotor_multi/obstacles/obstacles.py", [
        "该文件负责障碍物的生成、维护和局部障碍观测构造，是论文障碍避让实验的重要环境组成部分。",
        "它输出的不只是几何体本身，还包括供策略网络消费的 obstacle observation 切片和碰撞判定所需的地图状态。",
    ]),
    ("gym_art/quadrotor_multi/obstacles/utils.py", [
        "该文件提供障碍物观测与空间采样的辅助函数，尤其负责把周围几何布局转换成固定维度的局部 SDF 风格特征。",
        "这些工具函数的输出会直接拼进 agent 观测，因此分辨率、采样范围和排列顺序都决定了策略看到的障碍语义。",
    ]),
]

PATH_KEYWORDS = [
    ("runs/obstacles", [
        "该文件属于障碍场景实验配置，主要作用是把一组训练超参数打包成可复现实验入口。",
        "这些配置本身不执行仿真，但会控制环境难度、观测结构、回放概率和模型结构选择。",
    ]),
    ("runs/single_quad", [
        "该文件属于单机实验配置，用于验证基础飞行控制与奖励设计，而不是完整的多机协同避障。",
        "其参数会流入训练入口，决定单机环境、模型和日志行为。",
    ]),
    ("paper/", [
        "该文件用于论文结果分析或作图，消费训练日志、评估统计或中间结果来生成图表。",
        "它不改变训练流程，但决定如何把实验结果重新组织成论文中的可视化证据。",
    ]),
    ("sim2real", [
        "该文件属于 sim2real/部署链路，负责把训练得到的策略结构压缩、导出或验证到更接近机载执行的形式。",
        "这里处理的数据通常来自训练好的模型权重，下游会流向 C/torch 推理模块或部署测试。",
    ]),
    ("scenarios/", [
        "该文件定义环境场景或目标生成逻辑，决定每个 episode 如何布置目标、轨迹或队形任务。",
        "场景模块本身不负责控制，但它产生的初始条件会直接改变训练样本分布。",
    ]),
    ("collisions/", [
        "该文件处理机体、障碍物或房间边界的碰撞几何与碰撞后状态更新，是训练中安全相关奖励和终止判定的重要来源。",
        "这里的输出会回流到动力学状态、奖励项和碰撞统计中。",
    ]),
]

SYMBOL_HINTS = {
    "num_agents": "该值来自实验配置，决定环境一次并行维护多少架无人机；后续会影响观测拼接尺寸、邻居筛选范围和碰撞矩阵规模。",
    "num_use_neighbor_obs": "这个数量由 `neighbor_visible_num` 和无人机总数共同确定，表示每架无人机最终保留多少个邻居槽位给策略网络消费。",
    "obs_self_size": "这里先根据 `obs_repr` 查出单机自观测长度，后续才能正确切分 self observation、neighbor observation 和 obstacle observation。",
    "envs": "这个列表保存每架无人机各自的 `QuadrotorSingle` 实例；多机环境后续从这里汇总物理状态、执行动作并收集奖励。",
    "observation_space": "多机环境直接复用单机环境定义好的观测空间边界，确保训练侧看到的 shape 与底层观测编码保持一致。",
    "action_space": "动作空间沿用单机定义，因为策略仍然对每架无人机分别输出控制量，多机只是把多个 agent 并排组织起来。",
    "quad_arm": "机臂长度来自单机动力学参数，后续会被用作碰撞阈值缩放的物理参考尺度。",
    "control_freq": "控制频率由单机环境提供，表示策略动作被物理仿真执行的频率；后面的时间窗和 grace period 都会换算成这个步长。",
    "control_dt": "这个时间步长由控制频率反推得到，后续凡是按秒定义的持续时间都会依赖它折算成离散 step。",
    "pos": "该矩阵汇总所有无人机当前世界坐标，数据源是每个单机环境的动力学状态；后续用于构造相对位置、邻居排序和碰撞检测。",
    "vel": "该矩阵汇总所有无人机当前线速度，后续既用于邻居相对速度观测，也用于碰撞后速度修正和奖励计算。",
    "omega": "这里缓存所有无人机的角速度状态，主要服务于姿态稳定相关奖励、诊断日志和可能的观测拼接。",
    "rel_pos": "这个三维张量保存任意两架无人机之间的相对位移，是邻居观测、距离排序和机间碰撞判断的直接输入。",
    "rel_vel": "这个三维张量保存任意两架无人机之间的相对速度，用来告诉策略邻居是在接近还是远离，同时也辅助平滑碰撞惩罚。",
    "rew_coeff": "奖励权重表先从环境默认值起步，再叠加实验配置的覆盖项；后续每个 reward term 都按这里的系数折算成最终标量奖励。",
    "neighbor_obs_size": "该值根据邻居观测类型查询得到，决定每个邻居槽位在观测向量中占多少维。",
    "clip_neighbor_space_length": "这里先算出所有邻居观测总长度，后面才能从完整 observation space 中切出对应的上下界用于裁剪。",
    "use_obstacles": "这个开关来自实验配置，决定当前 episode 是否启用障碍物生成、障碍观测拼接和障碍碰撞逻辑。",
    "num_obstacles": "障碍物数量由密度和生成区域面积共同推导得到，目的是把抽象难度参数映射成场景中的实际障碍个数。",
    "scenario": "场景对象根据 `quads_mode` 创建，用来在 reset 时生成目标与初始布局，因此它直接控制训练样本分布。",
    "collisions_per_episode": "该计数器累计当前 episode 中发生的机间碰撞次数，既用于训练诊断，也可作为回放与评估指标。",
    "collisions_after_settle": "这里忽略出生阶段的瞬时重叠，只统计稳定飞行后的碰撞，以避免初始化噪声污染训练指标。",
    "collision_threshold": "实际碰撞阈值不是裸半径，而是按机臂长度缩放得到，确保不同尺寸动力学参数下碰撞判定仍有物理一致性。",
    "use_replay_buffer": "该开关控制是否启用碰撞片段回放机制；启用后环境会额外保存和恢复高风险状态片段。",
    "activate_replay_buffer": "回放不会一开始就启用，而是等策略学会基本飞行后再打开，避免早期随机行为把 buffer 污染成无意义样本。",
    "saved_in_replay_buffer": "这个标志用来防止同一次碰撞在 replay 模式下被重复回收，否则 buffer 会被重复失败片段淹没。",
    "use_numba": "该开关决定某些数值计算是否走 numba 优化路径，目的是在多机训练下减少物理仿真开销。",
    "use_downwash": "该开关决定是否模拟下洗气流干扰；打开后，垂直相对位置会影响相邻无人机的受力和控制难度。",
    "render_mode": "渲染模式决定环境输出给可视化层的方式；训练时通常关闭或简化，评估时才会真正消费这些渲染状态。",
    "distance_to_goal": "这个日志缓存按 agent 记录每一步到目标的距离，用于训练后分析轨迹质量和收敛过程。",
    "agent_col_agent": "该数组按 agent 记录机间碰撞状态或统计量，后续可写入 episode info 供训练日志系统聚合。",
    "agent_col_obst": "该数组按 agent 记录障碍碰撞状态或统计量，用于区分机间碰撞与障碍碰撞两类失败来源。",
    "parser": "命令行解析器先收集 Sample Factory 通用参数，再被四旋翼环境追加项目专用参数。",
    "partial_cfg": "这个中间配置只包含第一阶段解析结果，目的是先知道环境类型，再为该环境补充专属参数定义。",
    "final_cfg": "最终配置把通用参数和四旋翼专用参数合并到一起，后续训练、评估和环境构造都只消费这一份配置。",
    "cfg": "这里拿到的是训练或评估全流程共享的总配置对象，后续模型注册、环境创建和 PPO 超参数都会从中读取。",
    "status": "训练返回状态汇总了 Sample Factory 主循环的执行结果，调用方用它判断本次任务是否正常结束。",
}

FUNCTION_HINTS = {
    "register_swarm_components": [
        "这里把四旋翼项目需要的环境和模型注册进 Sample Factory 的全局注册表。",
        "只有完成这一步，后面的配置解析和训练循环才能通过字符串名称找到对应实现。",
    ],
    "parse_swarm_cfg": [
        "这个函数把 Sample Factory 的两阶段参数解析和四旋翼专属参数拼接起来。",
        "先拿到基础配置，再根据环境类型补充项目参数，最后生成供训练主循环直接消费的完整配置。",
    ],
    "main": [
        "这里串起训练脚本的顶层执行顺序：注册组件、解析配置、启动 RL 主循环。",
        "如果任一步缺失，训练入口就无法把论文里的实验配置落到实际环境和模型上。",
    ],
    "__init__": [
        "初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。",
        "这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。",
    ],
    "all_dynamics": [
        "这里把所有单机动力学对象打包成一个元组，方便上层模块批量访问底层物理状态。",
    ],
    "get_rel_pos_vel_item": [
        "这个辅助函数从全局位置/速度缓存中抽取某一架无人机相对于指定邻居集合的位移和速度差。",
        "输出会被进一步编码成邻居观测，因此它是从物理状态到策略输入的中间桥梁。",
    ],
    "get_obs_neighbor_rel": [
        "这里把相对位置和相对速度拼成单个邻居观测块，供策略网络判断邻居是否接近、远离或存在碰撞风险。",
    ],
    "extend_obs_space": [
        "该函数把单机原始观测扩展成带邻居信息的多机观测，并按 observation space 边界做裁剪。",
        "裁剪的目的是保证训练时输入范围稳定，不让极端相对距离破坏网络的尺度假设。",
    ],
    "neighborhood_indices": [
        "这个函数为每架无人机挑出当前应该暴露给策略网络的邻居索引集合。",
        "当可见邻居数受限时，它会根据相对距离裁剪邻居列表，把有限的观测容量留给最相关的近邻。",
    ],
}

RHS_HINTS = [
    ("parse_sf_args(", "这里先走 Sample Factory 的第一阶段解析，目的是得到基础配置和环境名称，再决定要补哪些四旋翼专用参数。"),
    ("parse_full_cfg(", "这里完成第二阶段解析，把新增的四旋翼参数真正写入最终配置对象。"),
    ('register_env("quadrotor_multi"', "这里把字符串环境名绑定到四旋翼环境工厂，训练配置里只要写 `quadrotor_multi` 就能实例化该环境。"),
    ("run_rl(", "这里正式进入 Sample Factory 的训练主循环，后续 rollout、优势估计、PPO 更新和 checkpoint 都由框架接管。"),
    ("QuadrotorSingle(", "这里为每个 agent 创建独立的单机环境和动力学状态，后续多机环境只是把这些单元在同一场景里协调起来。"),
    ("create_scenario(", "这里根据配置选择场景生成器，不同场景会改变目标分布、初始布局和任务难度。"),
    ("np.zeros([self.num_agents, 3])", "这里预分配按 agent 排列的三维物理状态缓存，后续每个 step 都会把单机状态同步到这块连续数组里。"),
    ("np.zeros((self.num_agents, self.num_agents, 3))", "这里预分配两两 agent 之间的关系张量，用空间换时间，避免每次构造邻居观测时重复申请内存。"),
    ("copy.deepcopy(self.rew_coeff)", "先保留一份默认奖励权重快照，后面可以验证外部覆盖参数是否合法，也方便区分默认值和实验改写项。"),
    ("dict(", "这里构造的是环境默认奖励权重表，表示在没有实验覆盖时多机导航任务各个目标项的基准权重。"),
    ("np.stack(", "这里把逐 agent 收集到的状态或观测重新压成批量张量，方便后续统一裁剪、拼接或送入网络。"),
    ("np.concatenate(", "这里执行观测拼接，把分散的物理特征重组为策略网络期望的固定顺序向量。"),
    ("np.clip(", "这里按 observation space 上下界裁剪邻居观测，避免极端数值破坏网络训练时的输入尺度。"),
    ("deque(", "这里使用定长队列保存最近若干 episode 的统计量，目的是观察训练趋势而不是无限累积历史。"),
]


def normalize_rel_path(rel_path: Path) -> str:
    return rel_path.as_posix()


def get_module_context(rel_path: Path) -> list[str]:
    rel = normalize_rel_path(rel_path)
    for path, desc in MODULE_DESCRIPTIONS:
        if rel == path:
            return desc
    for keyword, desc in PATH_KEYWORDS:
        if keyword in rel:
            return desc
    if rel.startswith("gym_art/quadrotor_multi/"):
        return [
            "该文件属于多机四旋翼仿真环境的一部分，负责环境状态、物理过程或配套工具中的某一环。",
            "它的上游通常来自场景配置、动力学状态或训练动作，下游会流向观测构造、奖励结算、碰撞处理或可视化。",
        ]
    if rel.startswith("swarm_rl/"):
        return [
            "该文件属于强化学习训练侧逻辑，负责把环境、模型、配置或评估流程接到 Sample Factory 框架上。",
            "这里产生的数据通常会继续流向训练循环、策略网络或实验分析脚本。",
        ]
    return [
        "该文件是项目代码库的一部分，当前副本的注释重点放在它在整体训练/仿真链路中的职责，而不是 Python 语法本身。",
    ]


def infer_assignment_hint(stripped: str) -> str | None:
    match = re.match(r"([A-Za-z_][A-Za-z0-9_\.\[\]'\"]*)\s*=", stripped)
    if match:
        target = clean_identifier(match.group(1))
        if target in SYMBOL_HINTS:
            return SYMBOL_HINTS[target]
    for key, hint in RHS_HINTS:
        if key in stripped:
            return hint
    return None


def infer_call_hint(stripped: str) -> str | None:
    for key, hint in RHS_HINTS:
        if key in stripped:
            return hint
    return None


def describe_control_flow(stripped: str) -> str | None:
    if stripped.startswith("if neighbor_visible_num == -1"):
        return "当配置要求“看到全部邻居”时，这里把邻居槽位数扩展为除自身外的所有无人机数量。"
    if stripped.startswith("if self.use_obstacles"):
        return "只有启用障碍场景时，下面这组状态才有意义，因为它们都服务于障碍生成、障碍观测和障碍碰撞统计。"
    if stripped.startswith("for i in range(self.num_agents)"):
        return "下面按 agent 逐个初始化或收集状态，目的是把单机物理单元拼成多机环境的批量结构。"
    if stripped.startswith("for key in self.rew_coeff.keys()"):
        return "这里把奖励权重统一转换成浮点数，避免命令行或配置覆盖带来的类型不一致影响后续奖励计算。"
    if stripped.startswith("if rew_coeff is not None"):
        return "如果实验配置覆盖了默认奖励权重，这里会先校验键是否合法，再把覆盖项写回环境内部的奖励表。"
    if stripped.startswith("if indices is None"):
        return "当调用方没有显式指定邻居集合时，这里默认把除自身外的所有无人机都纳入相对状态计算。"
    if stripped.startswith("elif 1 <= self.num_use_neighbor_obs < self.num_agents - 1"):
        return "当邻居槽位受限时，后续逻辑会从全部邻居里筛出最相关的一部分，而不是把所有相对状态都暴露给策略。"
    return None


def describe_line(stripped: str, rel_path: Path) -> list[str]:
    if not stripped or stripped.startswith("#"):
        return []
    if stripped.startswith(("import ", "from ")):
        return []
    if stripped in {'"""', "'''"}:
        return ["下面开始文件或代码块自带的文档字符串；如果源码作者已经解释设计意图，应优先结合它理解上下文。"]
    if stripped.startswith(('"""', "'''")):
        return ["下面的文档字符串通常由源码作者提供，用来补充模块职责、输入输出约束或使用方式。"]
    if stripped.startswith("@"):
        return ["这里通过装饰器把额外框架语义附着到下面的定义上，真正影响的是后续调用方式或注册行为。"]

    class_match = re.match(r"class\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
    if class_match:
        name = class_match.group(1)
        lines = [f"`{name}` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。"] 
        if name == "QuadrotorEnvMulti":
            lines.append("在这个项目里，它把多架 `QuadrotorSingle`、场景、障碍物和碰撞逻辑封装成符合 Gym/Sample Factory 接口的多智能体环境。")
        return lines

    func_match = re.match(r"(async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
    if func_match:
        name = func_match.group(2)
        return FUNCTION_HINTS.get(name, [f"`{name}` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。"])

    flow_hint = describe_control_flow(stripped)
    if flow_hint:
        return [flow_hint]

    assign_hint = infer_assignment_hint(stripped)
    if assign_hint:
        return [assign_hint]

    if stripped.startswith("return"):
        return ["这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。"]
    if stripped.startswith("assert"):
        return ["这里不是业务逻辑本身，而是在守护运行假设，避免非法配置或异常状态把后续训练流程带偏。"]
    if stripped.startswith(("if ", "elif ", "else:", "for ", "while ", "with ", "try:", "except", "finally:")):
        return []

    call_hint = infer_call_hint(stripped)
    if call_hint:
        return [call_hint]

    return []


def collect_python_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*.py"):
        rel_parts = path.relative_to(REPO_ROOT).parts
        if any(part in EXCLUDED_DIRS for part in rel_parts):
            continue
        files.append(path)
    return sorted(files)


def leading_spaces(text: str) -> str:
    return text[: len(text) - len(text.lstrip(" "))]


def strip_inline_comment(text: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    for idx, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if ch == "#" and not in_single and not in_double:
            return text[:idx]
    return text


def bracket_delta(text: str) -> int:
    code = strip_inline_comment(text)
    return sum(code.count(ch) for ch in "([{") - sum(code.count(ch) for ch in ")]}")


def clean_identifier(name: str) -> str:
    name = name.strip()
    name = name.replace("self.", "")
    name = name.replace("'", "")
    name = name.replace('"', "")
    return name


def split_targets(lhs: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in lhs:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(depth - 1, 0)
        if ch == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(ch)
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def assignment_comment(line: str) -> str | None:
    if "==" in line or "!=" in line or ">=" in line or "<=" in line:
        return None
    if line.startswith(("if ", "elif ", "while ", "assert ")):
        return None
    if line.strip().startswith(("*", "**")):
        return None

    op_match = re.search(r"(\+=|-=|\*=|/=|//=|%=|\*\*=|=)", line)
    if not op_match:
        return None

    lhs = line[: op_match.start()].strip()
    if not lhs:
        return None
    if lhs.endswith(","):
        lhs = lhs[:-1]
    names = [clean_identifier(part) for part in split_targets(lhs)]
    names = [name for name in names if name]
    if not names:
        return None
    if len(names) == 1:
        return f"保存或更新 `{names[0]}` 的值。"
    joined = "`, `".join(names[:4])
    return f"同时更新 `{joined}` 等变量。"


def call_comment(line: str) -> str | None:
    call_match = re.match(r"([A-Za-z_][A-Za-z0-9_\.]*)\(", line)
    if call_match:
        callee = call_match.group(1).split(".")[-1]
        return f"调用 `{callee}` 执行当前处理。"
    return None


def annotate_file(path: Path) -> str:
    rel_path = path.relative_to(REPO_ROOT)
    lines = path.read_text(encoding="utf-8").splitlines()
    output: list[str] = [
        f"# 中文注释副本；原始文件：{rel_path.as_posix()}",
        "# 说明：为避免修改源码，本文件仅作为阅读辅助材料。",
    ]
    for desc in get_module_context(rel_path):
        output.append(f"# {desc}")
    output.append("")

    in_import_block = False
    paren_depth = 0
    in_triple_string = False
    triple_delim = ""

    for line in lines:
        stripped = line.strip()
        indent = leading_spaces(line)
        is_import = stripped.startswith(("import ", "from "))
        triple_starts = (
            stripped.startswith('"""')
            or stripped.startswith("'''")
            or '"""' in stripped
            or "'''" in stripped
        )

        if is_import and not in_import_block:
            output.append(f"{indent}# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。")
        in_import_block = is_import

        skip_because_multiline = paren_depth > 0 and not stripped.endswith(":")
        should_comment = (
            not in_triple_string
            and not is_import
            and stripped
            and not stripped.startswith("#")
            and not skip_because_multiline
        )
        if should_comment:
            comments = describe_line(stripped, rel_path)
            for comment in comments:
                output.append(f"{indent}# {comment}")

        output.append(line)

        if not in_triple_string and triple_starts:
            if stripped.count('"""') % 2 == 1:
                in_triple_string = True
                triple_delim = '"""'
            elif stripped.count("'''") % 2 == 1:
                in_triple_string = True
                triple_delim = "'''"
        elif in_triple_string and triple_delim and triple_delim in stripped:
            if stripped.count(triple_delim) % 2 == 1:
                in_triple_string = False
                triple_delim = ""

        paren_depth += bracket_delta(line)
        if paren_depth < 0:
            paren_depth = 0

    return "\n".join(output) + "\n"


def build_directory_stats(files: list[Path]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for path in files:
        rel = path.relative_to(REPO_ROOT)
        key = "repo_root" if rel.parent == Path(".") else rel.parts[0]
        counter[key] += 1
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def build_inventory(files: list[Path]) -> str:
    lines = []
    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        annotated = (OUTPUT_DIR / rel).relative_to(REPO_ROOT).as_posix()
        lines.append(f"| `{rel}` | 已生成 | `{annotated}` |")
    return "\n".join(lines)


def build_summary(files: list[Path]) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats_lines = "\n".join(f"| `{group}` | {count} |" for group, count in build_directory_stats(files))
    inventory = build_inventory(files)

    return f"""# 项目中文注释进度与论文复现源码说明

## 1. 生成结果总览

- 生成时间：`{timestamp}`
- 论文说明文件：`{PAPER_PATH.relative_to(REPO_ROOT).as_posix()}`
- Python 文件总数：`{len(files)}`
- 已完成中文注释副本数量：`{len(files)}/{len(files)}`
- 注释输出目录：`annotated_python/`
- 说明：所有中文注释都写入独立副本，原始源码未被修改。

## 2. 注释进度统计

### 2.1 按目录统计

| 目录 | Python 文件数 |
| --- | ---: |
{stats_lines}

### 2.2 处理策略

- 注释从“解释语法动作”改为“解释工程语义”，优先说明数据来源、状态变更原因、下游用途和对训练/仿真的影响。
- 对环境状态、奖励系数、观测切片、碰撞统计、回放缓存、模型输入输出等关键对象重点解释；不再对每个赋值机械地写“保存或更新变量”。
- 对 `import`、简单样板代码和低信息密度片段，采用块级说明或直接留白，避免注释把真实逻辑淹没。
- 注释副本保留原始代码顺序，便于和源码逐行对照，同时允许通过文件头说明先交代模块在项目中的职责。

### 2.3 注释规范

- 文件头必须说明模块在整个四旋翼强化学习项目中的职责，以及它的上游输入和下游消费者。
- 类与函数注释优先解释“它维护什么状态/它完成哪段流程/结果交给谁继续使用”，而不是复述定义语法。
- 关键变量只在首次出现时注释，并明确数据来源、物理或训练含义、以及它在后续奖励、观测、碰撞、日志或 replay 中的用途。
- 对观测拼接、奖励整形、邻居筛选、障碍 SDF、注意力编码器、经验回放等论文关键机制，注释必须贴合本项目语境。
- 禁止使用“导入当前模块依赖”“调用某函数执行当前处理”“保存或更新变量值”这类无法帮助理解系统行为的浅层表述。

## 3. 论文《Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning》对应源码说明

### 3.1 训练入口与实验配置

- `swarm_rl/train.py`：训练主入口。负责注册 `quadrotor_multi` 环境、注册自定义模型，并调用 Sample Factory 的 `run_rl` 开始 APPO/IPPO 训练流程。
- `swarm_rl/env_wrappers/quadrotor_params.py`：集中定义论文实验会用到的命令行参数，包括无人机数量、邻居观测、障碍物观测、回放概率、碰撞奖励、房间大小等。
- `swarm_rl/runs/obstacles/quad_obstacle_baseline.py`：障碍环境基线配置，给出论文主实验的默认超参数，包括 `replay_buffer_sample_prob=0.75`、`quads_use_obstacles=True`、`quads_obstacle_obs_type=octomap`、`quads_use_downwash=True` 等。
- `swarm_rl/runs/obstacles/quads_multi_obstacles.py`：论文最终多机障碍实验配置。在基线之上启用 `quads_neighbor_visible_num=2`、`quads_neighbor_obs_type=pos_vel` 和 `quads_encoder_type=attention`，对应论文里的邻居感知和注意力网络。

### 3.2 环境、状态、动作与奖励

- `gym_art/quadrotor_multi/quadrotor_multi.py`：多机环境核心实现。负责多架无人机实例化、邻居选择、障碍生成、碰撞检测、奖励计算、回合重置、日志统计以及渲染调度。
- `gym_art/quadrotor_multi/quadrotor_single.py`：单架无人机底层环境与观测空间定义，包含自状态、姿态、角速度、房间相关观测等。
- `gym_art/quadrotor_multi/quadrotor_dynamics.py`、`quadrotor_control.py`、`sensor_noise.py`：动力学、控制和传感噪声实现，是论文“直接输出四旋翼推力”的物理基础。
- `swarm_rl/env_wrappers/reward_shaping.py`、`swarm_rl/env_wrappers/quad_utils.py`：把奖励系数、退火策略、环境包装和 Sample Factory 接口接起来；其中 `quadcol_bin`、`quadcol_bin_smooth_max`、`quadcol_bin_obst` 分别对应机间碰撞、平滑接近惩罚和障碍碰撞惩罚。

### 3.3 论文中的 SDF 障碍观测

- `gym_art/quadrotor_multi/obstacles/obstacles.py`：`MultiObstacles` 在 reset/step 时给每架无人机拼接 9 维障碍观测。
- `gym_art/quadrotor_multi/obstacles/utils.py`：`get_surround_sdfs` 用 `3x3` 局部网格和 `0.1m` 分辨率计算最近障碍距离，正对应论文里描述的 9 维、数量与排列无关的 SDF 风格障碍观测。
- `gym_art/quadrotor_multi/collisions/obstacles.py`：障碍碰撞后的法向与速度更新逻辑。

### 3.4 论文中的注意力模型

- `swarm_rl/models/quad_multi_model.py`：核心网络定义。
- `QuadMultiHeadAttentionEncoder`：当 `quads_encoder_type=attention` 时启用。它先分别编码 self / neighbor / obstacle 三类观测，再把 neighbor 和 obstacle embedding 拼成长度为 2 的 token 序列，送入多头注意力，再与 self embedding 融合输出。
- `QuadMultiEncoder`：非 attention 分支，对应论文早期/对照结构，使用 MLP 或其他邻居编码器。
- `swarm_rl/models/attention_layer.py`：多头注意力与单头注意力算子；其中单头版本用于 `sim2real` 小模型部署。

### 3.5 论文中的回放机制

- `gym_art/quadrotor_multi/quad_experience_replay.py`：论文回放机制的直接实现。
- 核心思路是：每隔一段时间保存环境检查点；一旦发生独特碰撞，就把“碰撞前约 1.5 秒”的状态存进 replay buffer；新 episode 开始时按概率采样这些历史碰撞片段重新训练。
- 这与论文中“放大碰撞事件、从裁剪后的碰撞片段继续训练”的描述一致，也是障碍密集环境下稳定训练的重要部分。

### 3.6 训练后分析、评估与复现实验辅助

- `swarm_rl/enjoy.py`：加载训练好的模型进行可视化评估。
- `swarm_rl/env_wrappers/v_value_map.py`、`paper/plot_v_value_*`：用于论文里的 V-value 可视化分析。
- `paper/*.py`：生成论文中的统计图、对比图和热力图。
- `swarm_rl/sim2real/`：面向部署的小模型与 C/torch 导出、测试逻辑，对应论文后半部分的板载部署讨论。

## 4. 训练流程与对应文件

1. 启动训练时，从 `swarm_rl/train.py` 进入，先注册环境和模型。
2. `parse_swarm_cfg()` 会结合 `swarm_rl/env_wrappers/quadrotor_params.py` 解析所有论文相关超参数。
3. `swarm_rl/env_wrappers/quad_utils.py` 根据配置构造 `QuadrotorEnvMulti`，并可选叠加 `ExperienceReplayWrapper`、奖励退火包装器和兼容层。
4. `gym_art/quadrotor_multi/quadrotor_multi.py` 在每个 step 中完成：动作施加、无人机动力学更新、机间/障碍/房间碰撞检测、奖励计算、障碍 SDF 观测拼接、统计信息更新。
5. `swarm_rl/models/quad_multi_model.py` 定义策略/价值网络的编码器；若启用论文主模型，则使用 `QuadMultiHeadAttentionEncoder`。
6. Sample Factory 的 APPO 训练循环根据这些环境与模型输出，完成 rollout、优势估计、PPO 更新、checkpoint 保存等工作。
7. 训练完成后，可用 `swarm_rl/enjoy.py` 做回放评估，也可用 `paper/*.py` 脚本生成论文中的统计图。

## 5. 论文复现时建议重点阅读的模块

- 环境主循环：`gym_art/quadrotor_multi/quadrotor_multi.py`
- 单机观测与动作接口：`gym_art/quadrotor_multi/quadrotor_single.py`
- 障碍观测与碰撞：`gym_art/quadrotor_multi/obstacles/obstacles.py`、`gym_art/quadrotor_multi/obstacles/utils.py`、`gym_art/quadrotor_multi/collisions/obstacles.py`
- 训练包装与奖励：`swarm_rl/env_wrappers/quad_utils.py`、`swarm_rl/env_wrappers/reward_shaping.py`、`swarm_rl/env_wrappers/quadrotor_params.py`
- 注意力模型：`swarm_rl/models/quad_multi_model.py`、`swarm_rl/models/attention_layer.py`
- 回放机制：`gym_art/quadrotor_multi/quad_experience_replay.py`
- 论文实验配置：`swarm_rl/runs/obstacles/quad_obstacle_baseline.py`、`swarm_rl/runs/obstacles/quads_multi_obstacles.py`

## 6. 注释副本清单

| 原始文件 | 状态 | 注释副本 |
| --- | --- | --- |
{inventory}
"""


def main() -> None:
    raise SystemExit(
        "Deprecated: do not use template-based bulk annotation generation. "
        "Use the manual file-by-file workflow documented in ANNOTATION_AND_PAPER_GUIDE.md."
    )


if __name__ == "__main__":
    main()
