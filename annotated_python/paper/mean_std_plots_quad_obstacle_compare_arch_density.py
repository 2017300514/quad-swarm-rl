# 中文注释副本；原始文件：paper/mean_std_plots_quad_obstacle_compare_arch_density.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import argparse
import os
import pickle
import sys
from os.path import join
from pathlib import Path

# 导入当前模块依赖。
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 导入当前模块依赖。
from sample_factory.utils.utils import ensure_dir_exists
from matplotlib.ticker import FuncFormatter

# 保存或更新 `PAGE_WIDTH_INCHES` 的值。
PAGE_WIDTH_INCHES = 8.2
# 保存或更新 `FULL_PAGE_WIDTH` 的值。
FULL_PAGE_WIDTH = 1.4 * PAGE_WIDTH_INCHES
# 保存或更新 `HALF_PAGE_WIDTH` 的值。
HALF_PAGE_WIDTH = FULL_PAGE_WIDTH / 2

# 保存或更新 `plt.rcParams[figure.figsize]` 的值。
plt.rcParams['figure.figsize'] = (FULL_PAGE_WIDTH, 2.5)  # (2.5, 2.0) 7.5， 4

# 保存或更新 `NUM_AGENTS` 的值。
NUM_AGENTS = 8
# 保存或更新 `EPISODE_DURATION` 的值。
EPISODE_DURATION = 16  # seconds
# 保存或更新 `TIME_METRIC_COLLISION` 的值。
TIME_METRIC_COLLISION = 60  # ONE MINUTE
# 保存或更新 `COLLISIONS_SCALE` 的值。
COLLISIONS_SCALE = ((
                            TIME_METRIC_COLLISION / EPISODE_DURATION) / NUM_AGENTS) * 2  # times two because 1 collision = 2 drones collided
# 保存或更新 `COLLISIONS_OBST_SCALE` 的值。
COLLISIONS_OBST_SCALE = ((
                                 TIME_METRIC_COLLISION / EPISODE_DURATION) / NUM_AGENTS)  # Not times two because 1 collision = 1 drone collide with 1 obstacle, and we only talk about drones here

# 保存或更新 `CRASH_GROUND_SCALE` 的值。
CRASH_GROUND_SCALE = (-1.0 / EPISODE_DURATION)

# 保存或更新 `PLOTS` 的值。
PLOTS = [
    dict(key='metric/agent_success_rate', name='Agent success rate', label='Average rate'),
    # dict(key='policy_stats/avg_distance_to_goal_1s', name='Avg. distance to the goal', label='Avg. distance(m)', clip_min=0.1, y_scale_formater=[0.1, 0.5, 1.0, 2.0]),
    # dict(key='metric/agent_deadlock_rate', name='Agent deadlock rate', label='Deadlock Rate'),
    dict(key='metric/agent_col_rate', name='Agent collision rate', label='Average rate'),
    # dict(key='metric/agent_obst_col_rate', name='Agent collision w/ obstacles rate'),
    dict(key='o_random/distance_to_goal_1s', name='Distance to goal (random)', label='Distance (m)'),
    dict(key='o_static_same_goal/distance_to_goal_1s', name='Distance to goal (same goal)', label='Distance (m)'),

]

# 保存或更新 `PLOT_STEP` 的值。
PLOT_STEP = int(5e6)
# 保存或更新 `TOTAL_STEP` 的值。
TOTAL_STEP = int(1e9 + 10000)

# 'blue': '#1F77B4', 'orange': '#FF7F0E', 'green': '#2CA02C', 'red': '#d70000'
# COLOR = ['#1F77B4', '#FF7F0E', '#2CA02C', '#d70000']
# 保存或更新 `COLOR` 的值。
COLOR = ['#4477AA', '#EE6677', '#54B345', '#CCBB44', '#AA3377']


# 定义函数 `extract`。
def extract(experiments):
    # 保存或更新 `scalar_accumulators` 的值。
    scalar_accumulators = [EventAccumulator(experiment_dir).Reload().scalars for experiment_dir in experiments]

    # Filter non event files
    # 保存或更新 `scalar_accumulators` 的值。
    scalar_accumulators = [scalar_accumulator for scalar_accumulator in scalar_accumulators if
                           scalar_accumulator.Keys()]

    # Get and validate all scalar keys
    # 保存或更新 `all_keys` 的值。
    all_keys = [tuple(sorted(scalar_accumulator.Keys())) for scalar_accumulator in scalar_accumulators]
    # assert len(set(all_keys)) == 1, \
    #     "All runs need to have the same scalar keys. There are mismatches in {}".format(all_keys)

    # 保存或更新 `keys` 的值。
    keys = all_keys[0]
    # 保存或更新 `all_scalar_events_per_key` 的值。
    all_scalar_events_per_key = [[scalar_accumulator.Items(key)
                                  for scalar_accumulator in scalar_accumulators] for key in keys]

    # Get and validate all steps per key
    # 保存或更新 `x_per_key` 的值。
    x_per_key = [[tuple(scalar_event.step
                        for scalar_event in scalar_events) for scalar_events in all_scalar_events]
                 for all_scalar_events in all_scalar_events_per_key]

    # 保存或更新 `plot_step` 的值。
    plot_step = PLOT_STEP
    # 保存或更新 `all_steps_per_key` 的值。
    all_steps_per_key = [[tuple(int(step_id) for step_id in range(0, TOTAL_STEP, plot_step))
                          for _ in all_scalar_events]
                         for all_scalar_events in all_scalar_events_per_key]

    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for i, all_steps in enumerate(all_steps_per_key):
        # 断言当前条件成立，用于保护运行假设。
        assert len(set(
            all_steps)) == 1, "For scalar {} the step numbering or count doesn't match. Step count for all runs: {}".format(
            keys[i], [len(steps) for steps in all_steps])

    # 保存或更新 `steps_per_key` 的值。
    steps_per_key = [all_steps[0] for all_steps in all_steps_per_key]

    # Get values per step per key
    # 保存或更新 `values_per_key` 的值。
    values_per_key = [[[scalar_event.value for scalar_event in scalar_events] for scalar_events in all_scalar_events]
                      for all_scalar_events in all_scalar_events_per_key]

    # 保存或更新 `interpolated_keys` 的值。
    interpolated_keys = dict()
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for tmp_id in range(len(PLOTS)):
        # 保存或更新 `key_idx` 的值。
        key_idx = keys.index(PLOTS[tmp_id]['key'])
        # 保存或更新 `values` 的值。
        values = values_per_key[key_idx]

        # 保存或更新 `x` 的值。
        x = steps_per_key[key_idx]
        # 保存或更新 `x_steps` 的值。
        x_steps = x_per_key[key_idx]

        # 保存或更新 `interpolated_y` 的值。
        interpolated_y = [[] for _ in values]

        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i in range(len(values)):
            # 保存或更新 `idx` 的值。
            idx = 0

            # 保存或更新 `tmp_min_step` 的值。
            tmp_min_step = min(len(x_steps[i]), len(values[i]))
            # 保存或更新 `values[i]` 的值。
            values[i] = values[i][2: tmp_min_step]
            # 保存或更新 `x_steps[i]` 的值。
            x_steps[i] = x_steps[i][2: tmp_min_step]

            # 断言当前条件成立，用于保护运行假设。
            assert len(x_steps[i]) == len(values[i])
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for x_idx in x:
                # 在条件成立时持续执行下面的循环体。
                while idx < len(x_steps[i]) - 1 and x_steps[i][idx] < x_idx:
                    # 保存或更新 `idx` 的值。
                    idx += 1

                # 根据条件决定是否进入当前分支。
                if x_idx == 0:
                    # 保存或更新 `interpolated_value` 的值。
                    interpolated_value = values[i][idx]
                # 当上一分支不满足时，继续判断新的条件。
                elif idx < len(values[i]) - 1:
                    # 保存或更新 `interpolated_value` 的值。
                    interpolated_value = (values[i][idx] + values[i][idx + 1]) / 2
                # 当前置条件都不满足时，执行兜底分支。
                else:
                    # 保存或更新 `interpolated_value` 的值。
                    interpolated_value = values[i][idx]

                # 执行这一行逻辑。
                interpolated_y[i].append(interpolated_value)
            # 断言当前条件成立，用于保护运行假设。
            assert len(interpolated_y[i]) == len(x)

        # 调用 `print` 执行当前处理。
        print(PLOTS[tmp_id]['key'], interpolated_y[0][:30])

        # 保存或更新 `interpolated_keys[PLOTS[tmp_id][key]]` 的值。
        interpolated_keys[PLOTS[tmp_id]['key']] = (x, interpolated_y)

    # 返回当前函数的结果。
    return interpolated_keys


# 定义函数 `aggregate`。
def aggregate(path, subpath, experiments, ax, legend_name, group_id):
    # 调用 `print` 执行当前处理。
    print("Started aggregation {}".format(path))

    # 保存或更新 `curr_dir` 的值。
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # 保存或更新 `cache_dir` 的值。
    cache_dir = join(curr_dir, 'cache')
    # 保存或更新 `cache_env` 的值。
    cache_env = join(cache_dir, subpath)

    # 根据条件决定是否进入当前分支。
    if os.path.isdir(cache_env):
        # 尝试执行下面的逻辑，并为异常情况做准备。
        try:
            # 使用上下文管理器包裹后续资源操作。
            with open(join(cache_env, f'{subpath}.pickle'), 'rb') as fobj:
                # 保存或更新 `interpolated_keys` 的值。
                interpolated_keys = pickle.load(fobj)

        # 捕获前面代码可能抛出的异常。
        except FileNotFoundError:
            # 保存或更新 `interpolated_keys` 的值。
            interpolated_keys = extract(experiments=experiments)

    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 保存或更新 `cache_env` 的值。
        cache_env = ensure_dir_exists(cache_env)
        # 保存或更新 `interpolated_keys` 的值。
        interpolated_keys = extract(experiments=experiments)
        # 使用上下文管理器包裹后续资源操作。
        with open(join(cache_env, f'{subpath}.pickle'), 'wb') as fobj:
            # 调用 `dump` 执行当前处理。
            pickle.dump(interpolated_keys, fobj)

    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for i, key in enumerate(interpolated_keys.keys()):
        # if i == (len(interpolated_keys.keys()) - 1) // 2:
        #     set_xlabel = True
        # else:
        #     set_xlabel = False
        # 保存或更新 `set_xlabel` 的值。
        set_xlabel = True
        # 调用 `plot` 执行当前处理。
        plot(i, interpolated_keys[key], ax[i], set_xlabel, legend_name, group_id)


# 定义函数 `smooth`。
def smooth(y, radius, mode='two_sided', valid_only=False):
    # 下面开始文档字符串说明。
    '''
    Smooth signal y, where radius is determines the size of the window

    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]

    valid_only: put nan in entries where the full-sized window is not available

    '''
    # 断言当前条件成立，用于保护运行假设。
    assert mode in ('two_sided', 'causal')
    # 根据条件决定是否进入当前分支。
    if len(y) < 2 * radius + 1:
        # 返回当前函数的结果。
        return np.ones_like(y) * y.mean()
    # 当上一分支不满足时，继续判断新的条件。
    elif mode == 'two_sided':
        # 保存或更新 `convkernel` 的值。
        convkernel = np.ones(2 * radius + 1)
        # 保存或更新 `out` 的值。
        out = np.convolve(y, convkernel, mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        # 根据条件决定是否进入当前分支。
        if valid_only:
            # 保存或更新 `out[:radius]` 的值。
            out[:radius] = out[-radius:] = np.nan
    # 当上一分支不满足时，继续判断新的条件。
    elif mode == 'causal':
        # 保存或更新 `convkernel` 的值。
        convkernel = np.ones(radius)
        # 保存或更新 `out` 的值。
        out = np.convolve(y, convkernel, mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        # 保存或更新 `out` 的值。
        out = out[:-radius + 1]
        # 根据条件决定是否进入当前分支。
        if valid_only:
            # 保存或更新 `out[:radius]` 的值。
            out[:radius] = np.nan
    # 返回当前函数的结果。
    return out


# 定义函数 `one_sided_ema`。
def one_sided_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    # 下面开始文档字符串说明。
    '''
    perform one-sided (causal) EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]

    Arguments:

    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds

    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]

    n: int                - number of points in new x grid

    decay_steps: float    - EMA decay factor, expressed in new x grid steps.

    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN

    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid

    '''

    # 保存或更新 `low` 的值。
    low = xolds[0] if low is None else low
    # 保存或更新 `high` 的值。
    high = xolds[-1] if high is None else high

    # 断言当前条件成立，用于保护运行假设。
    assert xolds[0] <= low, 'low = {} < xolds[0] = {} - extrapolation not permitted!'.format(low, xolds[0])
    # 断言当前条件成立，用于保护运行假设。
    assert xolds[-1] >= high, 'high = {} > xolds[-1] = {}  - extrapolation not permitted!'.format(high, xolds[-1])
    # 断言当前条件成立，用于保护运行假设。
    assert len(xolds) == len(yolds), 'length of xolds ({}) and yolds ({}) do not match!'.format(len(xolds), len(yolds))

    # 保存或更新 `xolds` 的值。
    xolds = xolds.astype('float64')
    # 保存或更新 `yolds` 的值。
    yolds = yolds.astype('float64')

    # 保存或更新 `luoi` 的值。
    luoi = 0  # last unused old index
    # 保存或更新 `sum_y` 的值。
    sum_y = 0.
    # 保存或更新 `count_y` 的值。
    count_y = 0.
    # 保存或更新 `xnews` 的值。
    xnews = np.linspace(low, high, n)
    # 保存或更新 `decay_period` 的值。
    decay_period = (high - low) / (n - 1) * decay_steps
    # 保存或更新 `interstep_decay` 的值。
    interstep_decay = np.exp(- 1. / decay_steps)
    # 保存或更新 `sum_ys` 的值。
    sum_ys = np.zeros_like(xnews)
    # 保存或更新 `count_ys` 的值。
    count_ys = np.zeros_like(xnews)
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for i in range(n):
        # 保存或更新 `xnew` 的值。
        xnew = xnews[i]
        # 保存或更新 `sum_y` 的值。
        sum_y *= interstep_decay
        # 保存或更新 `count_y` 的值。
        count_y *= interstep_decay
        # 在条件成立时持续执行下面的循环体。
        while True:
            # 根据条件决定是否进入当前分支。
            if luoi >= len(xolds):
                # 提前结束当前循环。
                break
            # 保存或更新 `xold` 的值。
            xold = xolds[luoi]
            # 根据条件决定是否进入当前分支。
            if xold <= xnew:
                # 保存或更新 `decay` 的值。
                decay = np.exp(- (xnew - xold) / decay_period)
                # 保存或更新 `sum_y` 的值。
                sum_y += decay * yolds[luoi]
                # 保存或更新 `count_y` 的值。
                count_y += decay
                # 保存或更新 `luoi` 的值。
                luoi += 1
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # 提前结束当前循环。
                break
        # 保存或更新 `sum_ys[i]` 的值。
        sum_ys[i] = sum_y
        # 保存或更新 `count_ys[i]` 的值。
        count_ys[i] = count_y

    # 保存或更新 `ys` 的值。
    ys = sum_ys / count_ys
    # 保存或更新 `ys[count_ys < low_counts_threshold]` 的值。
    ys[count_ys < low_counts_threshold] = np.nan

    # 返回当前函数的结果。
    return xnews, ys, count_ys


# 定义函数 `symmetric_ema`。
def symmetric_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    # 下面开始文档字符串说明。
    """
    perform symmetric EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]

    Arguments:

    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds

    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]

    n: int                - number of points in new x grid

    decay_steps: float    - EMA decay factor, expressed in new x grid steps.

    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN

    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid

    """
    # 同时更新 `xs`, `ys`, `count_ys` 等变量。
    xs, ys, count_ys = one_sided_ema(xolds, yolds, low, high, n, decay_steps, low_counts_threshold=0)
    # _, ys2, count_ys2 = one_sided_ema(-xolds[::-1], yolds[::-1], -high, -low, n, decay_steps, low_counts_threshold=0)
    # ys2 = ys2[::-1]
    # count_ys2 = count_ys2[::-1]
    # count_ys = count_ys1 + count_ys2
    # ys = (ys1 * count_ys1 + ys2 * count_ys2) / count_ys
    # ys[count_ys < low_counts_threshold] = np.nan
    # 返回当前函数的结果。
    return xs, ys, count_ys


# def plot(env, key, interpolated_key, ax, count):
# 定义函数 `plot`。
def plot(index, interpolated_key, ax, set_xlabel, legend_name, group_id):
    # 保存或更新 `params` 的值。
    params = PLOTS[index]

    # set title
    # 保存或更新 `title_text` 的值。
    title_text = params['name']
    # 保存或更新 `ax.set_title(title_text, fontsize` 的值。
    ax.set_title(title_text, fontsize=10)
    # 根据条件决定是否进入当前分支。
    if set_xlabel:
        # 调用 `set_xlabel` 执行当前处理。
        ax.set_xlabel('Total environment steps')

    # 同时更新 `x`, `y` 等变量。
    x, y = interpolated_key
    # 保存或更新 `x` 的值。
    x = np.array(x)
    # 保存或更新 `y_np` 的值。
    y_np = [np.array(yi) for yi in y]
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for i, yi in enumerate(y_np):
        # 同时更新 `xnew`, `y_np[i]`, `_` 等变量。
        xnew, y_np[i], _ = symmetric_ema(x, yi, x[0], x[-1], n=200, decay_steps=1.0)

    # 保存或更新 `x` 的值。
    x = xnew

    # 保存或更新 `y_np` 的值。
    y_np = [smooth(yi, 1) for yi in y_np]
    # 保存或更新 `y_np` 的值。
    y_np = np.stack(y_np)

    # 保存或更新 `logscale` 的值。
    logscale = params.get('logscale', False)
    # 根据条件决定是否进入当前分支。
    if logscale:
        # 保存或更新 `ax.set_yscale(log, base` 的值。
        ax.set_yscale('log', base=2)
        # 调用 `set_minor_locator` 执行当前处理。
        ax.yaxis.set_minor_locator(ticker.NullLocator())  # no minor ticks

    # 定义函数 `scientific`。
    def scientific(x, pos):
        # x:  tick value - ie. what you currently see in yticks
        # pos: a position - ie. the index of the tick (from 0 to 9 in this example)
        # 返回当前函数的结果。
        return '%.1f' % x

    # 保存或更新 `scientific_formatter` 的值。
    scientific_formatter = FuncFormatter(scientific)
    # 调用 `set_major_formatter` 执行当前处理。
    ax.yaxis.set_major_formatter(scientific_formatter)
    # TO CHANGE
    # 调用 `set_ylim` 执行当前处理。
    ax.set_ylim(-0.02, 1.0 + 0.02)
    # if np.max(y_np) < 0.6:
    #     ax.set_ylim(-0.02, 0.62)
    # elif np.max(y_np) < 0.8:
    #     ax.set_ylim(-0.02, 0.82)

    # ax.yaxis.set_major_formatter(ticker.ScalarFormatter())  # set regular formatting

    # 保存或更新 `coeff` 的值。
    coeff = params.get('coeff', 1.0)
    # 保存或更新 `y_np` 的值。
    y_np *= coeff

    # 保存或更新 `mutate` 的值。
    mutate = params.get('mutate', None)
    # 根据条件决定是否进入当前分支。
    if mutate:
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for i in range(y_np.shape[1]):
            # 保存或更新 `y_np[:, i]` 的值。
            y_np[:, i] = mutate(y_np[:, i])

    # 保存或更新 `y_mean` 的值。
    y_mean = np.mean(y_np, axis=0)
    # 保存或更新 `y_std` 的值。
    y_std = np.std(y_np, axis=0)
    # 保存或更新 `y_plus_std` 的值。
    y_plus_std = y_mean + y_std
    # 保存或更新 `y_minus_std` 的值。
    y_minus_std = y_mean - y_std

    # 保存或更新 `clip_max` 的值。
    clip_max = params.get('clip_max', None)
    # 根据条件决定是否进入当前分支。
    if clip_max:
        # 保存或更新 `y_mean` 的值。
        y_mean = np.minimum(y_mean, clip_max)
        # 保存或更新 `y_plus_std` 的值。
        y_plus_std = np.minimum(y_plus_std, clip_max)
        # 保存或更新 `y_minus_std` 的值。
        y_minus_std = np.minimum(y_minus_std, clip_max)

    # 保存或更新 `clip_min` 的值。
    clip_min = params.get('clip_min', None)
    # 根据条件决定是否进入当前分支。
    if clip_min:
        # 保存或更新 `y_mean` 的值。
        y_mean = np.maximum(y_mean, clip_min)
        # 保存或更新 `y_plus_std` 的值。
        y_plus_std = np.maximum(y_plus_std, clip_min)
        # 保存或更新 `y_minus_std` 的值。
        y_minus_std = np.maximum(y_minus_std, clip_min)

    # 定义函数 `mkfunc`。
    def mkfunc(x, pos):
        # 根据条件决定是否进入当前分支。
        if x >= 1e9:
            # 返回当前函数的结果。
            return '%dB' % int(x * 1e-9)
        # 当上一分支不满足时，继续判断新的条件。
        elif x >= 1e6:
            # 返回当前函数的结果。
            return '%dM' % int(x * 1e-6)
        # 当上一分支不满足时，继续判断新的条件。
        elif x >= 1e3:
            # 返回当前函数的结果。
            return '%dK' % int(x * 1e-3)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 返回当前函数的结果。
            return '%d' % int(x)

    # 保存或更新 `mkformatter` 的值。
    mkformatter = matplotlib.ticker.FuncFormatter(mkfunc)
    # 调用 `set_major_formatter` 执行当前处理。
    ax.xaxis.set_major_formatter(mkformatter)

    # 执行这一行逻辑。
    ax.spines['right'].set_visible(False)
    # 执行这一行逻辑。
    ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    # ax.spines['top'].set_linewidth(0.5)
    # ax.spines['right'].set_linewidth(0.5)
    # 执行这一行逻辑。
    ax.spines['left'].set_linewidth(0.6)
    # 执行这一行逻辑。
    ax.spines['bottom'].set_linewidth(0.6)

    # 保存或更新 `label` 的值。
    label = params.get('label')
    # 根据条件决定是否进入当前分支。
    if label:
        # 保存或更新 `ax.set_ylabel(label, fontsize` 的值。
        ax.set_ylabel(label, fontsize=10)

        # hide tick of axis
        # 调用 `tick_bottom` 执行当前处理。
        ax.xaxis.tick_bottom()
    # 调用 `tick_left` 执行当前处理。
    ax.yaxis.tick_left()
    # 保存或更新 `ax.tick_params(which` 的值。
    ax.tick_params(which='major', length=0)

    # 保存或更新 `ax.grid(color` 的值。
    ax.grid(color='#B3B3B3', linestyle='-', linewidth=0.25, alpha=0.2)
    # ax.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))

    # 保存或更新 `lw` 的值。
    lw = 1.4

    # 保存或更新 `ax.plot(x, y_mean, color` 的值。
    ax.plot(x, y_mean, color=COLOR[group_id], label=legend_name, linewidth=lw, antialiased=True)
    # 保存或更新 `ax.fill_between(x, y_minus_std, y_plus_std, color` 的值。
    ax.fill_between(x, y_minus_std, y_plus_std, color=COLOR[group_id], alpha=0.25)
    # ax.legend(prop={'size': 6}, loc='lower right')


# 定义函数 `hide_tick_spine`。
def hide_tick_spine(ax):
    # 执行这一行逻辑。
    ax.spines['right'].set_visible(False)
    # 执行这一行逻辑。
    ax.spines['top'].set_visible(False)
    # 执行这一行逻辑。
    ax.spines['left'].set_visible(False)
    # 保存或更新 `ax.tick_params(labelcolor` 的值。
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)


# 定义函数 `main`。
def main():
    # 保存或更新 `parser` 的值。
    parser = argparse.ArgumentParser()
    # 保存或更新 `parser.add_argument(--path, type` 的值。
    parser.add_argument('--path', type=str, help='main path for tensorboard files', default=os.getcwd())
    # 保存或更新 `parser.add_argument(--output, type` 的值。
    parser.add_argument('--output', type=str,
                        help='aggregation can be saved as tensorboard file (summary) or as table (csv)', default='csv')

    # 保存或更新 `args` 的值。
    args = parser.parse_args()
    # 保存或更新 `path` 的值。
    path = Path(args.path)

    # 根据条件决定是否进入当前分支。
    if not path.exists():
        # 主动抛出异常以中止或提示错误。
        raise argparse.ArgumentTypeError('Parameter {} is not a valid path'.format(path))

    # TO CHANGE
    # 保存或更新 `subpaths` 的值。
    subpaths = sorted(os.listdir(path))
    # subpaths = ['20%', '40%', '60%', '80%']
    # 保存或更新 `legend_name` 的值。
    legend_name = sorted(['20%', '40%', '60%', '80%'])
    # legend_name = sorted(['1', '2', '6', '16', '31'])
    # 保存或更新 `all_experiment_dirs` 的值。
    all_experiment_dirs = {}
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for subpath in subpaths:
        # 根据条件决定是否进入当前分支。
        if subpath not in all_experiment_dirs:
            # 保存或更新 `all_experiment_dirs[subpath]` 的值。
            all_experiment_dirs[subpath] = []

        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for filename in Path(args.path + "/" + subpath).rglob('*.tfevents.*'):
            # 保存或更新 `experiment_dir` 的值。
            experiment_dir = os.path.dirname(filename)
            # 执行这一行逻辑。
            all_experiment_dirs[subpath].append(experiment_dir)

    # 根据条件决定是否进入当前分支。
    if args.output not in ['summary', 'csv']:
        # 主动抛出异常以中止或提示错误。
        raise argparse.ArgumentTypeError("Parameter {} is not summary or csv".format(args.output))

    # TO CHANGE
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    # ax = (ax1, ax2, ax3, ax4)
    # 同时更新 `fig`, `(ax1, ax2, ax3, ax4)` 等变量。
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    # 保存或更新 `ax` 的值。
    ax = (ax1, ax2, ax3, ax4)


    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for i in range(len(all_experiment_dirs)):
        # 调用 `aggregate` 执行当前处理。
        aggregate(path, subpaths[i], all_experiment_dirs[subpaths[i]], ax, legend_name[i], i)

    # 同时更新 `handles`, `labels` 等变量。
    handles, labels = ax[0].get_legend_handles_labels()
    # TO CHANGE
    # 保存或更新 `lgd` 的值。
    lgd = fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.07), fontsize=10)
    # 调用 `set_in_layout` 执行当前处理。
    lgd.set_in_layout(True)

    # 保存或更新 `plt.tight_layout(pad` 的值。
    plt.tight_layout(pad=1.0)
    # 保存或更新 `plt.subplots_adjust(wspace` 的值。
    plt.subplots_adjust(wspace=0.15, hspace=0.25)
    # plt.margins(0, 0)

    # 保存或更新 `save_dir` 的值。
    save_dir = os.path.join(os.getcwd(), 'final_plots')
    # 根据条件决定是否进入当前分支。
    if not os.path.exists(save_dir):
        # 调用 `makedirs` 执行当前处理。
        os.makedirs(save_dir)

    # 保存或更新 `figname` 的值。
    figname = 'scale_obst_density.pdf'
    # 保存或更新 `plt.savefig(os.path.join(save_dir, figname), format` 的值。
    plt.savefig(os.path.join(save_dir, figname), format='pdf', bbox_inches='tight', pad_inches=0.01)

    # 返回当前函数的结果。
    return 0


# 根据条件决定是否进入当前分支。
if __name__ == '__main__':
    # 调用 `exit` 执行当前处理。
    sys.exit(main())
