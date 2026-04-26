# 中文注释副本；原始文件：gym_art/quadrotor_multi/numba_utils.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import numpy as np
import numpy.random as nr
from numba import njit, types, vectorize, int32, float32, double, boolean
from numba.core.errors import TypingError
from numba.extending import overload
from numba.experimental import jitclass


# 为下面的函数或方法附加装饰器行为。
@overload(np.clip)
# 定义函数 `impl_clip`。
def impl_clip(a, a_min, a_max):
    # Check that `a_min` and `a_max` are scalars, and at most one of them is None.
    # 根据条件决定是否进入当前分支。
    if not isinstance(a_min, (types.Integer, types.Float, types.NoneType)):
        # 主动抛出异常以中止或提示错误。
        raise TypingError("a_min must be a_min scalar int/float")
    # 根据条件决定是否进入当前分支。
    if not isinstance(a_max, (types.Integer, types.Float, types.NoneType)):
        # 主动抛出异常以中止或提示错误。
        raise TypingError("a_max must be a_min scalar int/float")
    # 根据条件决定是否进入当前分支。
    if isinstance(a_min, types.NoneType) and isinstance(a_max, types.NoneType):
        # 主动抛出异常以中止或提示错误。
        raise TypingError("a_min and a_max can't both be None")

    # 根据条件决定是否进入当前分支。
    if isinstance(a, (types.Integer, types.Float)):
        # `a` is a scalar with a valid type
        # 根据条件决定是否进入当前分支。
        if isinstance(a_min, types.NoneType):
            # `a_min` is None
            # 定义函数 `impl`。
            def impl(a, a_min, a_max):
                # 返回当前函数的结果。
                return min(a, a_max)
        # 当上一分支不满足时，继续判断新的条件。
        elif isinstance(a_max, types.NoneType):
            # `a_max` is None
            # 定义函数 `impl`。
            def impl(a, a_min, a_max):
                # 返回当前函数的结果。
                return max(a, a_min)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # neither `a_min` or `a_max` are None
            # 定义函数 `impl`。
            def impl(a, a_min, a_max):
                # 返回当前函数的结果。
                return min(max(a, a_min), a_max)
    # 当上一分支不满足时，继续判断新的条件。
    elif (
            isinstance(a, types.Array) and
            a.ndim == 1 and
            isinstance(a.dtype, (types.Integer, types.Float))
    # 这里开始一个新的代码块。
    ):
        # `a` is a 1D array of the proper type
        # 定义函数 `impl`。
        def impl(a, a_min, a_max):
            # Allocate an output array using standard numpy functions
            # 保存或更新 `out` 的值。
            out = np.empty_like(a)
            # Iterate over `a`, calling `np.clip` on every element
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for i in range(a.size):
                # This will dispatch to the proper scalar implementation (as
                # defined above) at *compile time*. There should have no
                # overhead at runtime.
                # 保存或更新 `out[i]` 的值。
                out[i] = np.clip(a[i], a_min, a_max)
            # 返回当前函数的结果。
            return out
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 主动抛出异常以中止或提示错误。
        raise TypingError("`a` must be an int/float or a 1D array of ints/floats")

    # The call to `np.clip` has arguments with valid types, return our
    # numba-compatible implementation
    # 返回当前函数的结果。
    return impl


# 为下面的函数或方法附加装饰器行为。
@vectorize(nopython=True)
# 定义函数 `angvel2thrust_numba`。
def angvel2thrust_numba(w, linearity=0.424):
    # 返回当前函数的结果。
    return (1 - linearity) * w ** 2 + linearity * w


# 为下面的函数或方法附加装饰器行为。
@njit
# 定义函数 `numba_cross`。
def numba_cross(a, b):
    # 返回当前函数的结果。
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])


# 保存或更新 `spec` 的值。
spec = [
    ('action_dimension', int32),
    ('mu', float32),
    ('theta', float32),
    ('sigma', float32),
    ('state', double[:]),
    ('use_seed', boolean)
]


# 为下面的函数或方法附加装饰器行为。
@jitclass(spec)
# 定义类 `OUNoiseNumba`。
class OUNoiseNumba:
    # 下面的文档字符串用于说明当前模块或代码块。
    """Ornstein–Uhlenbeck process"""

    # 定义函数 `__init__`。
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3, use_seed=False):
        # 下面开始文档字符串说明。
        """
        @param: mu: mean of noise
        @param: theta: stabilization coeff (i.e. noise return to mean)
        @param: sigma: noise scale coeff
        @param: use_seed: set the random number generator to some specific seed for test
        """
        # 保存或更新 `action_dimension` 的值。
        self.action_dimension = action_dimension
        # 保存或更新 `mu` 的值。
        self.mu = mu
        # 保存或更新 `theta` 的值。
        self.theta = theta
        # 保存或更新 `sigma` 的值。
        self.sigma = sigma
        # 保存或更新 `state` 的值。
        self.state = np.ones(self.action_dimension) * self.mu
        # 调用 `reset` 执行当前处理。
        self.reset()

        # 根据条件决定是否进入当前分支。
        if use_seed:
            # 调用 `seed` 执行当前处理。
            nr.seed(2)

    # 定义函数 `reset`。
    def reset(self):
        # 保存或更新 `state` 的值。
        self.state = np.ones(self.action_dimension) * self.mu

    # 定义函数 `noise`。
    def noise(self):
        # 保存或更新 `x` 的值。
        x = self.state
        # 保存或更新 `dx` 的值。
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        # 保存或更新 `state` 的值。
        self.state = x + dx
        # 返回当前函数的结果。
        return self.state

