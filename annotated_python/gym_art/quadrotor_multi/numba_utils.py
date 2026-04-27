# 中文注释副本；原始文件：gym_art/quadrotor_multi/numba_utils.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这个文件的职责是给动力学与控制主链补齐一层 “numba 能直接 JIT 的数值积木”。
# 上游调用者通常是电机推力换算、向量几何和探索噪声模块；下游则是 `quadrotor_dynamics.py`
# 或相关低层数值循环，目标是在不牺牲 Python 代码结构的前提下，把热点运算搬到 nopython 路径。

import numpy as np
import numpy.random as nr
from numba import njit, types, vectorize, int32, float32, double, boolean
from numba.core.errors import TypingError
from numba.extending import overload
from numba.experimental import jitclass


@overload(np.clip)
def impl_clip(a, a_min, a_max):
    # 这里不是重新定义 numpy API，而是给 numba 注册一个它能理解的 `np.clip` 版本。
    # 没有这层 overload 时，某些 jit 路径里对标量或 1D 向量的裁剪会因为类型推断失败而退回 Python。
    if not isinstance(a_min, (types.Integer, types.Float, types.NoneType)):
        raise TypingError("a_min must be a_min scalar int/float")
    if not isinstance(a_max, (types.Integer, types.Float, types.NoneType)):
        raise TypingError("a_max must be a_min scalar int/float")
    if isinstance(a_min, types.NoneType) and isinstance(a_max, types.NoneType):
        raise TypingError("a_min and a_max can't both be None")

    if isinstance(a, (types.Integer, types.Float)):
        if isinstance(a_min, types.NoneType):
            def impl(a, a_min, a_max):
                return min(a, a_max)
        elif isinstance(a_max, types.NoneType):
            def impl(a, a_min, a_max):
                return max(a, a_min)
        else:
            def impl(a, a_min, a_max):
                return min(max(a, a_min), a_max)
    elif (
        isinstance(a, types.Array) and
        a.ndim == 1 and
        isinstance(a.dtype, (types.Integer, types.Float))
    ):
        # 数组版本显式逐元素展开，这样 numba 会在编译期把内部 `np.clip` 分发到上面的标量实现。
        def impl(a, a_min, a_max):
            out = np.empty_like(a)
            for i in range(a.size):
                out[i] = np.clip(a[i], a_min, a_max)
            return out
    else:
        raise TypingError("`a` must be an int/float or a 1D array of ints/floats")

    return impl


@vectorize(nopython=True)
def angvel2thrust_numba(w, linearity=0.424):
    # 这就是电机角速度到推力的简化非线性映射；
    # 上层动力学会成批调用它，把每个 rotor 的命令转成物理推力。
    return (1 - linearity) * w ** 2 + linearity * w


@njit
def numba_cross(a, b):
    # 手写叉乘是为了让热点向量几何留在纯 numba 路径，不必频繁退回 numpy 的通用实现。
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


spec = [
    ('action_dimension', int32),
    ('mu', float32),
    ('theta', float32),
    ('sigma', float32),
    ('state', double[:]),
    ('use_seed', boolean)
]


@jitclass(spec)
class OUNoiseNumba:
    """Ornstein–Uhlenbeck process"""

    # 这是一个 numba 版 OU 噪声生成器。
    # 它维护连续时间相关的内部状态 `state`，适合给动作空间加“平滑抖动”，而不是每步完全独立的白噪声。
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3, use_seed=False):
        """
        @param: mu: mean of noise
        @param: theta: stabilization coeff (i.e. noise return to mean)
        @param: sigma: noise scale coeff
        @param: use_seed: set the random number generator to some specific seed for test
        """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

        if use_seed:
            nr.seed(2)

    def reset(self):
        # 新 episode 或新 rollout 开始时把噪声状态拉回均值，避免前一个片段的相关噪声直接串到下一个片段里。
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        # OU 更新把当前状态往均值 `mu` 拉回，同时再叠一层高斯扰动，
        # 因此输出会比独立高斯噪声更平滑、更像电机或控制量的连续抖动。
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state
