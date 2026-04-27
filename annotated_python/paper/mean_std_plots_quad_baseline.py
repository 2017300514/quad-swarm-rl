# 中文注释副本；原始文件：paper/mean_std_plots_quad_baseline.py
# 该脚本把无障碍 baseline 训练的 TensorBoard 日志聚合成论文图。
# 上游输入是多个训练 run 目录下的 `.tfevents.*` 标量；下游输出是 `quads_baseline.pdf`，
# 用来展示总回报、到目标距离、机间碰撞率和飞行时长占比随训练步数的变化。

import argparse
import os
import pickle
import sys
from os.path import join
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from plots.plot_utils import ORANGE, set_matplotlib_params
from sample_factory.utils.utils import ensure_dir_exists

set_matplotlib_params()

PAGE_WIDTH_INCHES = 8.2
FULL_PAGE_WIDTH = 1.4 * PAGE_WIDTH_INCHES
HALF_PAGE_WIDTH = FULL_PAGE_WIDTH / 2

plt.rcParams['figure.figsize'] = (FULL_PAGE_WIDTH, 2.3)

NUM_AGENTS = 8
EPISODE_DURATION = 16
TIME_METRIC_COLLISION = 60
COLLISIONS_SCALE = ((TIME_METRIC_COLLISION / EPISODE_DURATION) / NUM_AGENTS) * 2
CRASH_GROUND_SCALE = (-1.0 / EPISODE_DURATION)

# `PLOTS` 规定每个子图消费哪个 TensorBoard key，以及如何把原始奖励项换算成论文里的物理量。
PLOTS = [
    dict(key='0_aux/avg_reward', name='Total reward', label='Avg. episode reward'),
    dict(
        key='0_aux/avg_rewraw_pos',
        name='Avg. distance to the target',
        label='Avg. distance, meters',
        coeff=-1.0 / EPISODE_DURATION,
    ),
    dict(
        key='0_aux/avg_num_collisions_after_settle',
        name='Avg. collisions between drones per minute',
        label='Number of collisions',
        logscale=True,
        coeff=COLLISIONS_SCALE,
        clip_min=0.05,
    ),
    dict(
        key='0_aux/avg_rewraw_crash',
        name='Flight performance',
        label='Fraction of the episode in the air',
        coeff=CRASH_GROUND_SCALE,
        mutate=lambda y: 1 - y,
        clip_max=1.0,
    ),
]

PLOT_STEP = int(5e6)
TOTAL_STEP = int(1e9 + 10000)


def extract(experiments):
    # 每个 experiment 目录对应一条训练 run；这里统一读取它们的标量事件。
    scalar_accumulators = [EventAccumulator(experiment_dir).Reload().scalars for experiment_dir in experiments]

    # 排除空目录或非事件目录，避免把无效路径混进统计。
    scalar_accumulators = [
        scalar_accumulator for scalar_accumulator in scalar_accumulators if scalar_accumulator.Keys()
    ]

    # baseline 图要对多条 run 求均值，因此默认所有 run 拥有完全一致的标量 key 集合。
    all_keys = [tuple(sorted(scalar_accumulator.Keys())) for scalar_accumulator in scalar_accumulators]
    assert len(set(all_keys)) == 1, (
        "All runs need to have the same scalar keys. There are mismatches in {}".format(all_keys)
    )

    keys = all_keys[0]
    all_scalar_events_per_key = [
        [scalar_accumulator.Items(key) for scalar_accumulator in scalar_accumulators] for key in keys
    ]

    # 原始日志写入频率不一定严格对齐，先取出每条 run 的真实 step 轨迹。
    x_per_key = [
        [tuple(scalar_event.step for scalar_event in sorted(scalar_events)) for scalar_events in sorted(all_scalar_events)]
        for all_scalar_events in all_scalar_events_per_key
    ]

    # 再把所有 run 投影到统一的 5e6 step 网格，后续均值和标准差带才能直接比较。
    plot_step = PLOT_STEP
    all_steps_per_key = [
        [tuple(int(step_id) for step_id in range(0, TOTAL_STEP, plot_step)) for _ in sorted(all_scalar_events)]
        for all_scalar_events in all_scalar_events_per_key
    ]

    for i, all_steps in enumerate(all_steps_per_key):
        assert len(set(all_steps)) == 1, (
            "For scalar {} the step numbering or count doesn't match. Step count for all runs: {}".format(
                keys[i], [len(steps) for steps in all_steps]
            )
        )

    steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]

    values_per_key = [
        [[scalar_event.value for scalar_event in scalar_events] for scalar_events in all_scalar_events]
        for all_scalar_events in all_scalar_events_per_key
    ]

    interpolated_keys = dict()
    for plot_idx in range(len(PLOTS)):
        key_idx = keys.index(PLOTS[plot_idx]['key'])
        values = values_per_key[key_idx]
        x = steps_per_key[key_idx]
        x_steps = x_per_key[key_idx]

        interpolated_y = [[] for _ in values]

        for run_idx in range(len(values)):
            idx = 0

            # 丢掉最开头两个点，减少训练刚启动时日志不稳定对曲线起点的影响。
            tmp_min_step = min(len(x_steps[run_idx]), len(values[run_idx]))
            values[run_idx] = values[run_idx][2:tmp_min_step]
            x_steps[run_idx] = x_steps[run_idx][2:tmp_min_step]

            assert len(x_steps[run_idx]) == len(values[run_idx])
            for x_idx in x:
                while idx < len(x_steps[run_idx]) - 1 and x_steps[run_idx][idx] < x_idx:
                    idx += 1

                # 这里用近邻/局部均值的粗插值，把不规则日志对齐到固定论文步长。
                if x_idx == 0:
                    interpolated_value = values[run_idx][idx]
                elif idx < len(values[run_idx]) - 1:
                    interpolated_value = (values[run_idx][idx] + values[run_idx][idx + 1]) / 2
                else:
                    interpolated_value = values[run_idx][idx]

                interpolated_y[run_idx].append(interpolated_value)
            assert len(interpolated_y[run_idx]) == len(x)

        interpolated_keys[PLOTS[plot_idx]['key']] = (x, interpolated_y)

    return interpolated_keys


def aggregate(path, subpath, experiments, ax):
    # 聚合结果缓存到 `paper/cache/<subpath>/`，下次重复作图时不必再次重扫事件文件。
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = join(curr_dir, 'cache')
    cache_env = join(cache_dir, subpath)

    if os.path.isdir(cache_env):
        with open(join(cache_env, f'{subpath}.pickle'), 'rb') as fobj:
            interpolated_keys = pickle.load(fobj)
    else:
        cache_env = ensure_dir_exists(cache_env)
        interpolated_keys = extract(experiments=experiments)
        with open(join(cache_env, f'{subpath}.pickle'), 'wb') as fobj:
            pickle.dump(interpolated_keys, fobj)

    for i, key in enumerate(interpolated_keys.keys()):
        plot(i, interpolated_keys[key], ax[i])


def plot(index, interpolated_key, ax):
    params = PLOTS[index]

    ax.set_title(params['name'], fontsize=8)
    ax.set_xlabel('Simulation steps')

    x, y = interpolated_key
    y_np = [np.array(yi) for yi in y]
    y_np = np.stack(y_np)

    # 碰撞数跨数量级变化较大，因此用对数轴更容易看出收敛趋势。
    logscale = params.get('logscale', False)
    if logscale:
        ax.set_yscale('log', base=2)
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    # `coeff` 和 `mutate` 把训练日志里的奖励项还原成论文展示口径。
    coeff = params.get('coeff', 1.0)
    y_np *= coeff

    mutate = params.get('mutate', None)
    if mutate:
        for i in range(y_np.shape[1]):
            y_np[:, i] = mutate(y_np[:, i])

    y_mean = np.mean(y_np, axis=0)
    y_std = np.std(y_np, axis=0)
    y_plus_std = y_mean + y_std
    y_minus_std = y_mean - y_std

    clip_max = params.get('clip_max', None)
    if clip_max:
        y_mean = np.minimum(y_mean, clip_max)
        y_plus_std = np.minimum(y_plus_std, clip_max)
        y_minus_std = np.minimum(y_minus_std, clip_max)

    clip_min = params.get('clip_min', None)
    if clip_min:
        y_mean = np.maximum(y_mean, clip_min)
        y_plus_std = np.maximum(y_plus_std, clip_min)
        y_minus_std = np.maximum(y_minus_std, clip_min)

    # 训练步数很长，格式化成 K/M/B 便于论文图排版。
    def mkfunc(x, pos):
        if x >= 1e9:
            return '%dB' % int(x * 1e-9)
        elif x >= 1e6:
            return '%dM' % int(x * 1e-6)
        elif x >= 1e3:
            return '%dK' % int(x * 1e-3)
        else:
            return '%d' % int(x)

    mkformatter = matplotlib.ticker.FuncFormatter(mkfunc)
    ax.xaxis.set_major_formatter(mkformatter)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.0)

    label = params.get('label')
    if label:
        ax.set_ylabel(label, fontsize=8)
        ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.tick_params(which='major', length=0)

    ax.grid(color='#B3B3B3', linestyle='-', linewidth=0.25, alpha=0.2)

    lw = 1.4
    ax.plot(x, y_mean, color=ORANGE, linewidth=lw, antialiased=True)
    ax.fill_between(x, y_minus_std, y_plus_std, color=ORANGE, alpha=0.25, antialiased=True, linewidth=0.0)


def hide_tick_spine(ax):
    # 这是旧版排版残留的辅助函数，当前主流程没有实际调用。
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='main path for tensorboard files', default=os.getcwd())
    parser.add_argument(
        '--output',
        type=str,
        help='aggregation can be saves as tensorboard file (summary) or as table (csv)',
        default='csv',
    )

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        raise argparse.ArgumentTypeError('Parameter {} is not a valid path'.format(path))

    # baseline 图默认把输入路径下的所有 run 视作同一实验族，取第一个子目录名作为 cache key。
    subpath = os.listdir(path)[0]
    all_experiment_dirs = []
    for filename in Path(args.path).rglob('*.tfevents.*'):
        experiment_dir = os.path.dirname(filename)
        all_experiment_dirs.append(experiment_dir)

    if args.output not in ['summary', 'csv']:
        raise argparse.ArgumentTypeError("Parameter {} is not summary or csv".format(args.output))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax = (ax1, ax2, ax3, ax4)

    aggregate(path, subpath, all_experiment_dirs, ax=ax)

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    # 最终图写到 `../final_plots/`，供论文排版直接引用。
    plt.savefig(
        os.path.join(os.getcwd(), '../final_plots/quads_baseline.pdf'),
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.01,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
