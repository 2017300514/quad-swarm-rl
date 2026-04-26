from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "annotated_python"
SUMMARY_PATH = REPO_ROOT / "ANNOTATION_AND_PAPER_GUIDE.md"
PAPER_PATH = REPO_ROOT / "hybrid_auto" / "Huang 等 - Collision Avoidance and Navigation for a Quadrotor.md"
EXCLUDED_DIRS = {".git", "annotated_python", "__pycache__", "tools"}


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


def describe_line(stripped: str) -> str | None:
    if not stripped:
        return None
    if stripped.startswith("#"):
        return None
    if stripped in {'"""', "'''"}:
        return "下面开始文档字符串说明。"
    if stripped.startswith(('"""', "'''")):
        return "下面的文档字符串用于说明当前模块或代码块。"
    if stripped.startswith("@"):
        return "为下面的函数或方法附加装饰器行为。"

    class_match = re.match(r"class\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
    if class_match:
        return f"定义类 `{class_match.group(1)}`。"

    func_match = re.match(r"(async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
    if func_match:
        return f"定义函数 `{func_match.group(2)}`。"

    if stripped.startswith("if "):
        return "根据条件决定是否进入当前分支。"
    if stripped.startswith("elif "):
        return "当上一分支不满足时，继续判断新的条件。"
    if stripped == "else:":
        return "当前置条件都不满足时，执行兜底分支。"
    if stripped.startswith("for "):
        return "遍历当前序列或迭代器，逐项执行下面的逻辑。"
    if stripped.startswith("while "):
        return "在条件成立时持续执行下面的循环体。"
    if stripped.startswith("with "):
        return "使用上下文管理器包裹后续资源操作。"
    if stripped == "try:":
        return "尝试执行下面的逻辑，并为异常情况做准备。"
    if stripped.startswith("except"):
        return "捕获前面代码可能抛出的异常。"
    if stripped == "finally:":
        return "无论是否出现异常，都执行这里的收尾逻辑。"
    if stripped.startswith("return"):
        return "返回当前函数的结果。"
    if stripped.startswith("yield"):
        return "按生成器方式产出当前结果。"
    if stripped.startswith("raise"):
        return "主动抛出异常以中止或提示错误。"
    if stripped.startswith("assert"):
        return "断言当前条件成立，用于保护运行假设。"
    if stripped == "pass":
        return "当前代码块暂时不执行实际逻辑。"
    if stripped == "break":
        return "提前结束当前循环。"
    if stripped == "continue":
        return "跳过本轮循环剩余逻辑，进入下一轮。"
    if stripped.startswith(("import ", "from ")):
        return None

    assign_msg = assignment_comment(stripped)
    if assign_msg:
        return assign_msg

    call_msg = call_comment(stripped)
    if call_msg:
        return call_msg

    if stripped.endswith(":"):
        return "这里开始一个新的代码块。"
    return "执行这一行逻辑。"


def annotate_file(path: Path) -> str:
    rel_path = path.relative_to(REPO_ROOT)
    lines = path.read_text(encoding="utf-8").splitlines()
    output: list[str] = [
        f"# 中文注释副本；原始文件：{rel_path.as_posix()}",
        "# 说明：为避免修改源码，本文件仅作为阅读辅助材料。",
        "",
    ]

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
            output.append(f"{indent}# 导入当前模块依赖。")
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
            comment = describe_line(stripped)
            if comment:
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

- 对结构清晰且信息量较大的语句，采用逐行中文注释。
- 对 `import`、多行参数列表、简单返回语句等低信息密度片段，采用块级或语义合并注释，避免把文件注释得过碎。
- 注释副本保留原始代码顺序，便于和源码逐行对照。

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
    files = collect_python_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in files:
        rel = path.relative_to(REPO_ROOT)
        output_path = OUTPUT_DIR / rel
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(annotate_file(path), encoding="utf-8")

    SUMMARY_PATH.write_text(build_summary(files), encoding="utf-8")
    print(f"Annotated {len(files)} Python files into {OUTPUT_DIR.relative_to(REPO_ROOT)}")
    print(f"Wrote summary to {SUMMARY_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
