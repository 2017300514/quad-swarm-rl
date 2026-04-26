# 中文注释副本；原始文件：gym_art/quadrotor_multi/numba_utils.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 该文件属于多机四旋翼仿真环境的一部分，负责环境状态、物理过程或配套工具中的某一环。
# 它的上游通常来自场景配置、动力学状态或训练动作，下游会流向观测构造、奖励结算、碰撞处理或可视化。

# 下面这组导入把当前模块会消费的环境组件、训练接口或数值工具集中拉进来；真正重要的是后续它们怎样参与数据流。
import numpy as np
import numpy.random as nr
from numba import njit, types, vectorize, int32, float32, double, boolean
from numba.core.errors import TypingError
from numba.extending import overload
from numba.experimental import jitclass


# 这里通过装饰器把额外框架语义附着到下面的定义上，真正影响的是后续调用方式或注册行为。
@overload(np.clip)
# `impl_clip` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def impl_clip(a, a_min, a_max):
    # Check that `a_min` and `a_max` are scalars, and at most one of them is None.
    if not isinstance(a_min, (types.Integer, types.Float, types.NoneType)):
        raise TypingError("a_min must be a_min scalar int/float")
    if not isinstance(a_max, (types.Integer, types.Float, types.NoneType)):
        raise TypingError("a_max must be a_min scalar int/float")
    if isinstance(a_min, types.NoneType) and isinstance(a_max, types.NoneType):
        raise TypingError("a_min and a_max can't both be None")

    if isinstance(a, (types.Integer, types.Float)):
        # `a` is a scalar with a valid type
        if isinstance(a_min, types.NoneType):
            # `a_min` is None
            # `impl` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
            def impl(a, a_min, a_max):
                # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
                return min(a, a_max)
        elif isinstance(a_max, types.NoneType):
            # `a_max` is None
            # `impl` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
            def impl(a, a_min, a_max):
                # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
                return max(a, a_min)
        else:
            # neither `a_min` or `a_max` are None
            # `impl` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
            def impl(a, a_min, a_max):
                # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
                return min(max(a, a_min), a_max)
    elif (
            isinstance(a, types.Array) and
            a.ndim == 1 and
            isinstance(a.dtype, (types.Integer, types.Float))
    ):
        # `a` is a 1D array of the proper type
        # `impl` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
        def impl(a, a_min, a_max):
            # Allocate an output array using standard numpy functions
            out = np.empty_like(a)
            # Iterate over `a`, calling `np.clip` on every element
            for i in range(a.size):
                # This will dispatch to the proper scalar implementation (as
                # defined above) at *compile time*. There should have no
                # overhead at runtime.
                # 这里按 observation space 上下界裁剪邻居观测，避免极端数值破坏网络训练时的输入尺度。
                out[i] = np.clip(a[i], a_min, a_max)
            # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
            return out
    else:
        raise TypingError("`a` must be an int/float or a 1D array of ints/floats")

    # The call to `np.clip` has arguments with valid types, return our
    # numba-compatible implementation
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return impl


# 这里通过装饰器把额外框架语义附着到下面的定义上，真正影响的是后续调用方式或注册行为。
@vectorize(nopython=True)
# `angvel2thrust_numba` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def angvel2thrust_numba(w, linearity=0.424):
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return (1 - linearity) * w ** 2 + linearity * w


# 这里通过装饰器把额外框架语义附着到下面的定义上，真正影响的是后续调用方式或注册行为。
@njit
# `numba_cross` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
def numba_cross(a, b):
    # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])


spec = [
    ('action_dimension', int32),
    ('mu', float32),
    ('theta', float32),
    ('sigma', float32),
    ('state', double[:]),
    ('use_seed', boolean)
]


# 这里通过装饰器把额外框架语义附着到下面的定义上，真正影响的是后续调用方式或注册行为。
@jitclass(spec)
# `OUNoiseNumba` 是当前文件暴露的核心类型，它负责维护与该模块职责直接相关的长期状态。
class OUNoiseNumba:
    # 下面的文档字符串通常由源码作者提供，用来补充模块职责、输入输出约束或使用方式。
    """Ornstein–Uhlenbeck process"""

    # 初始化阶段会把实验配置翻译成环境内部状态，包括单机实例、观测裁剪边界、碰撞阈值、障碍物和日志缓存。
    # 这些状态会在后续每个 step 中被不断读取和更新，因此这里决定了环境运行时的数据布局。
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3, use_seed=False):
        # 下面开始文件或代码块自带的文档字符串；如果源码作者已经解释设计意图，应优先结合它理解上下文。
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

    # `reset` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    # `noise` 封装了当前模块中的一段独立流程，阅读时应重点关注它消费哪些状态、又把结果交给谁继续使用。
    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        # 这里把当前阶段整理好的结果交还给上层调用者；真正要理解的是返回值之后会进入哪条训练或仿真链路。
        return self.state

