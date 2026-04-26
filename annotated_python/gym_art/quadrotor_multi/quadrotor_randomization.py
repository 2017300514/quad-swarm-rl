# 中文注释副本；原始文件：gym_art/quadrotor_multi/quadrotor_randomization.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import copy
import numpy as np
from numpy.linalg import norm
from copy import deepcopy

# 导入当前模块依赖。
from gym_art.quadrotor_multi.quad_utils import *
from gym_art.quadrotor_multi.quad_models import *


# 定义函数 `clip_params_positive`。
def clip_params_positive(params):
    # 定义函数 `clip_positive`。
    def clip_positive(key, item):
        # 返回当前函数的结果。
        return np.clip(item, a_min=0., a_max=None)
    # 调用 `walk_dict` 执行当前处理。
    walk_dict(params, clip_positive)
    # 返回当前函数的结果。
    return params

# 定义函数 `check_quad_param_limits`。
def check_quad_param_limits(params, params_init=None):
    ## Body parameters (like lengths and masses) are always positive
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for key in ["body", "payload", "arms", "motors", "propellers"]:
        # 保存或更新 `params[geom][key]` 的值。
        params["geom"][key] = clip_params_positive(params["geom"][key])

    # 保存或更新 `params[geom][motor_pos][xyz][:2]` 的值。
    params["geom"]["motor_pos"]["xyz"][:2] = np.clip(params["geom"]["motor_pos"]["xyz"][:2], a_min=0.005, a_max=None)
    # 保存或更新 `body_w` 的值。
    body_w = params["geom"]["body"]["w"]
    # 保存或更新 `params[geom][payload_pos][xy]` 的值。
    params["geom"]["payload_pos"]["xy"] = np.clip(params["geom"]["payload_pos"]["xy"], a_min=-body_w/4., a_max=body_w/4.)    
    # 保存或更新 `params[geom][arms_pos][angle]` 的值。
    params["geom"]["arms_pos"]["angle"] = np.clip(params["geom"]["arms_pos"]["angle"], a_min=0., a_max=90.)    
    
    ## Damping parameters
    # 保存或更新 `params[damp][vel]` 的值。
    params["damp"]["vel"] = np.clip(params["damp"]["vel"], a_min=0.00000, a_max=1.)
    # 保存或更新 `params[damp][omega_quadratic]` 的值。
    params["damp"]["omega_quadratic"] = np.clip(params["damp"]["omega_quadratic"], a_min=0.00000, a_max=1.)
    
    ## Motor parameters
    # 保存或更新 `params[motor][thrust_to_weight]` 的值。
    params["motor"]["thrust_to_weight"] = np.clip(params["motor"]["thrust_to_weight"], a_min=1.2, a_max=None)
    # 保存或更新 `params[motor][torque_to_thrust]` 的值。
    params["motor"]["torque_to_thrust"] = np.clip(params["motor"]["torque_to_thrust"], a_min=0.001, a_max=1.)
    # 保存或更新 `params[motor][linearity]` 的值。
    params["motor"]["linearity"] = np.clip(params["motor"]["linearity"], a_min=0., a_max=1.)
    # 保存或更新 `params[motor][assymetry]` 的值。
    params["motor"]["assymetry"] = np.clip(params["motor"]["assymetry"], a_min=0.9, a_max=1.1)
    # 保存或更新 `params[motor][C_drag]` 的值。
    params["motor"]["C_drag"] = np.clip(params["motor"]["C_drag"], a_min=0., a_max=None)
    # 保存或更新 `params[motor][C_roll]` 的值。
    params["motor"]["C_roll"] = np.clip(params["motor"]["C_roll"], a_min=0., a_max=None)
    # 保存或更新 `params[motor][damp_time_up]` 的值。
    params["motor"]["damp_time_up"] = np.clip(params["motor"]["damp_time_up"], a_min=0., a_max=None)
    # 保存或更新 `params[motor][damp_time_down]` 的值。
    params["motor"]["damp_time_down"] = np.clip(params["motor"]["damp_time_down"], a_min=0., a_max=None)

    ## Make sure propellers make sense in size
    # 根据条件决定是否进入当前分支。
    if params_init is not None:
        # 保存或更新 `r0` 的值。
        r0 = params_init["geom"]["propellers"]["r"]
        # 同时更新 `t2w`, `t2w0` 等变量。
        t2w, t2w0 = params_init["motor"]["thrust_to_weight"], params["motor"]["thrust_to_weight"]
        # 保存或更新 `params[geom][propellers][r]` 的值。
        params["geom"]["propellers"]["r"] = r0 * (t2w/t2w0)**0.5

    # 返回当前函数的结果。
    return params

# 定义函数 `get_dyn_randomization_params`。
def get_dyn_randomization_params(quad_params, noise_ratio=0., noise_ratio_params=None):
    # 下面开始文档字符串说明。
    """
    The function updates noise params
    Args:
        noise_ratio (float): ratio of change relative to the nominal values
        noise_ratio_params (dict): if for some parameters you want to have different ratios relative to noise_ratio,
            you can provided it through this dictionary
    Returns:
        noise_params dictionary
    """
    ## Setting the initial noise ratios (nominal ones)
    # 保存或更新 `noise_params` 的值。
    noise_params = deepcopy(quad_params)
    # 定义函数 `set_noise_ratio`。
    def set_noise_ratio(key, item):
        # 根据条件决定是否进入当前分支。
        if isinstance(item, str):
            # 返回当前函数的结果。
            return None
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # 返回当前函数的结果。
            return noise_ratio
    
    # 调用 `walk_dict` 执行当前处理。
    walk_dict(noise_params, set_noise_ratio)

    ## Updating noise ratios
    # 根据条件决定是否进入当前分支。
    if noise_ratio_params is not None:
        # noise_params.update(noise_ratio_params)
        # 调用 `dict_update_existing` 执行当前处理。
        dict_update_existing(noise_params, noise_ratio_params)
    # 返回当前函数的结果。
    return noise_params


# 定义函数 `perturb_dyn_parameters`。
def perturb_dyn_parameters(params, noise_params, sampler="normal"):
    # 下面开始文档字符串说明。
    """
    The function samples around nominal parameters provided noise parameters
    Args:
        params (dict): dictionary of quadrotor parameters
        noise_params (dict): dictionary of noise parameters with the same hierarchy as params, but
            contains ratio of deviation from the params
    Returns:
        dict: modified parameters
    """
    ## Sampling parameters
    # 定义函数 `sample_normal`。
    def sample_normal(key, param_val, ratio):
        #2*ratio since 2std contain 98% of all samples
        # 保存或更新 `param_val_sample` 的值。
        param_val_sample = np.random.normal(loc=param_val, scale=np.abs((ratio/2)*np.array(param_val)))
        # 返回当前函数的结果。
        return param_val_sample, ratio
    
    # 定义函数 `sample_uniform`。
    def sample_uniform(key, param_val, ratio):
        # 保存或更新 `param_val` 的值。
        param_val = np.array(param_val)
        # 返回当前函数的结果。
        return np.random.uniform(low=param_val - param_val*ratio, high=param_val + param_val*ratio), ratio

    # 保存或更新 `sample_param` 的值。
    sample_param = locals()["sample_" + sampler]

    # 保存或更新 `params_new` 的值。
    params_new = deepcopy(params)
    # 调用 `walk_2dict` 执行当前处理。
    walk_2dict(params_new, noise_params, sample_param)

    ## Fixing a few parameters if they go out of allowed limits
    # 保存或更新 `params_new` 的值。
    params_new = check_quad_param_limits(params_new, params)
    # print_dic(params_new)

    # 返回当前函数的结果。
    return params_new

# 定义函数 `resample_dyn_parameters`。
def resample_dyn_parameters(params, noise_params, sampler="uniform"):
    # 下面开始文档字符串说明。
    """
    The function resamples dynamics parameters
    Args:
        params (dict): dictionary of quadrotor parameters
        noise_params (dict): dictionary of noise parameters with the same hierarchy as params, but
            contains ratio of deviation from the params
    Returns:
        dict: modified parameters
    """
    ## Sampling parameters
    # 定义函数 `sample_normal`。
    def sample_normal(key, param_val, min_max):
        #2*ratio since 2std contain 98% of all samples
        # 保存或更新 `mean` 的值。
        mean = (min_max.min + min_max.max) / 2
        # 保存或更新 `std` 的值。
        std = (min_max.max - min_max.min) / 4 # i.e. 2 * stds contain 98% of samples
        # 返回当前函数的结果。
        return np.random.normal(
                loc=mean, scale=std
            )
    
    # 定义函数 `sample_uniform`。
    def sample_uniform(key, param_val, min_max):
        # 返回当前函数的结果。
        return np.random.uniform(
            low=min_max.min * np.ones_like(param_val), 
            high=min_max.max * np.ones_like(param_val)
        )

    # 保存或更新 `sample_param` 的值。
    sample_param = locals()["sample_" + sampler]

    # 保存或更新 `params_new` 的值。
    params_new = deepcopy(params)
    # 调用 `walk_2dict` 执行当前处理。
    walk_2dict(params_new, noise_params, sample_param)

    ## Fixing a few parameters if they go out of allowed limits
    # 保存或更新 `params_new` 的值。
    params_new = check_quad_param_limits(params_new, params)

    # 返回当前函数的结果。
    return params_new


# 定义函数 `randomquad_parameters`。
def randomquad_parameters():
    # 下面开始文档字符串说明。
    """
    The function samples parameters for all possible quadrotors
    Args:
        scale (float): scale of sampling
    Returns:
        dict: sampled quadrotor parameters
    """
    ###################################################################
    ## DENSITIES (body, payload, arms, motors, propellers)
    # Crazyflie estimated body / payload / arms / motors / props density: 1388.9 / 1785.7 / 1777.8 / 1948.8 / 246.6 kg/m^3
    # Hummingbird estimated body / payload / arms / motors/ props density: 588.2 / 173.6 / 1111.1 / 509.3 / 246.6 kg/m^3
    # 保存或更新 `geom_params` 的值。
    geom_params = {}
    # 保存或更新 `dens_val` 的值。
    dens_val = np.random.uniform(
        low=[500., 200., 500., 500., 200.], 
        high=[2000., 2000., 2000., 4500., 300.])
    
    # 保存或更新 `geom_params[body]` 的值。
    geom_params["body"] = {"density": dens_val[0]}
    # 保存或更新 `geom_params[payload]` 的值。
    geom_params["payload"] = {"density": dens_val[1]}
    # 保存或更新 `geom_params[arms]` 的值。
    geom_params["arms"] = {"density": dens_val[2]}
    # 保存或更新 `geom_params[motors]` 的值。
    geom_params["motors"] = {"density": dens_val[3]}
    # 保存或更新 `geom_params[propellers]` 的值。
    geom_params["propellers"] = {"density": dens_val[4]}

    ###################################################################
    ## GEOMETRIES
    # MOTORS (and overal size)
    # 保存或更新 `total_w` 的值。
    total_w = np.random.uniform(low=0.05, high=0.2)
    # 保存或更新 `total_l` 的值。
    total_l = np.clip(np.random.normal(loc=1., scale=0.1), a_min=1.0, a_max=None) * total_w
    # 保存或更新 `motor_z` 的值。
    motor_z = np.random.normal(loc=0., scale=total_w / 8.)
    # 保存或更新 `geom_params[motor_pos]` 的值。
    geom_params["motor_pos"] = {"xyz": [total_w / 2., total_l / 2., motor_z]}
    # 保存或更新 `geom_params[motors][r]` 的值。
    geom_params["motors"]["r"] = total_w * np.random.normal(loc=0.1, scale=0.01)
    # 保存或更新 `geom_params[motors][h]` 的值。
    geom_params["motors"]["h"] = geom_params["motors"]["r"] * np.random.normal(loc=1.0, scale=0.05)
    
    # BODY
    # 同时更新 `w_low`, `w_high` 等变量。
    w_low, w_high = 0.25, 0.5
    # 保存或更新 `w_coeff` 的值。
    w_coeff = np.random.uniform(low=w_low, high=w_high)
    # 保存或更新 `geom_params[body][w]` 的值。
    geom_params["body"]["w"] = w_coeff * total_w
    ## Promotes more elangeted bodies when they are more narrow
    # 保存或更新 `l_scale` 的值。
    l_scale = (1. - (w_coeff - w_low) / (w_high - w_low))
    # 保存或更新 `geom_params[body][l]` 的值。
    geom_params["body"]["l"] =  np.clip(np.random.normal(loc=1., scale=l_scale), a_min=1.0, a_max=None) * geom_params["body"]["w"]
    # 保存或更新 `geom_params[body][h]` 的值。
    geom_params["body"]["h"] =  np.random.uniform(low=0.1, high=1.5) * geom_params["body"]["w"]

    # PAYLOAD
    # 保存或更新 `pl_scl` 的值。
    pl_scl = np.random.uniform(low=0.25, high=1.0, size=3)
    # 保存或更新 `geom_params[payload][w]` 的值。
    geom_params["payload"]["w"] =  pl_scl[0] * geom_params["body"]["w"]
    # 保存或更新 `geom_params[payload][l]` 的值。
    geom_params["payload"]["l"] =  pl_scl[1] * geom_params["body"]["l"]
    # 保存或更新 `geom_params[payload][h]` 的值。
    geom_params["payload"]["h"] =  pl_scl[2] * geom_params["body"]["h"]
    # 保存或更新 `geom_params[payload_pos]` 的值。
    geom_params["payload_pos"] = {
            "xy": np.random.normal(loc=0., scale=geom_params["body"]["w"] / 10., size=2), 
            "z_sign": np.sign(np.random.uniform(low=-1, high=1))}
    # z_sing corresponds to location (+1 - on top of the body, -1 - on the bottom of the body)

    # ARMS
    # 保存或更新 `geom_params[arms][w]` 的值。
    geom_params["arms"]["w"] = total_w * np.random.normal(loc=0.05, scale=0.005)
    # 保存或更新 `geom_params[arms][h]` 的值。
    geom_params["arms"]["h"] = total_w * np.random.normal(loc=0.05, scale=0.005)
    # 保存或更新 `geom_params[arms_pos]` 的值。
    geom_params["arms_pos"] = {"angle": np.random.normal(loc=45., scale=10.), "z": motor_z - geom_params["motors"]["h"]/2.}
    
    # PROPS
    # 保存或更新 `thrust_to_weight` 的值。
    thrust_to_weight = np.random.uniform(low=1.5, high=3.5)
    # thrust_to_weight = np.random.uniform(low=1.8, high=2.5)
    # 保存或更新 `geom_params[propellers][h]` 的值。
    geom_params["propellers"]["h"] = 0.01
    # 保存或更新 `geom_params[propellers][r]` 的值。
    geom_params["propellers"]["r"] = (0.3) * total_w * (thrust_to_weight / 2.0)**0.5
    
    ## Damping parameters
    # damp_vel_scale = np.random.uniform(low=0.01, high=2.)
    # damp_omega_scale = damp_vel_scale * np.random.uniform(low=0.75, high=1.25)
    # damp_params = {
    #     "vel": 0.001 * damp_vel_scale, 
    #     "omega_quadratic": 0.015 * damp_omega_scale}
    # 保存或更新 `damp_params` 的值。
    damp_params = {
        "vel": 0.0, 
        "omega_quadratic": 0.0}

    ## Noise parameters
    # 保存或更新 `noise_params` 的值。
    noise_params = {}
    # 保存或更新 `noise_params[thrust_noise_ratio]` 的值。
    noise_params["thrust_noise_ratio"] = np.random.uniform(low=0.01, high=0.05) #0.01
    
    ## Motor parameters
    # 保存或更新 `damp_time_up` 的值。
    damp_time_up = np.random.uniform(low=0.15, high=0.2)
    # 保存或更新 `damp_time_down_scale` 的值。
    damp_time_down_scale = np.random.uniform(low=1.0, high=1.0)
    # 保存或更新 `motor_params` 的值。
    motor_params = {"thrust_to_weight" : thrust_to_weight,
                    "torque_to_thrust": np.random.uniform(low=0.005, high=0.025), #0.05 originally
                    "assymetry": np.random.uniform(low=0.9, high=1.1, size=4),
                    "linearity": 1.0,
                    "C_drag": 0.,
                    "C_roll": 0.,
                    "damp_time_up": damp_time_up,
                    "damp_time_down": damp_time_down_scale * damp_time_up
                    # "linearity": np.random.normal(loc=0.5, scale=0.1)
                    }

    ## Summarizing
    # 保存或更新 `params` 的值。
    params = {
        "geom": geom_params, 
        "damp": damp_params, 
        "noise": noise_params,
        "motor": motor_params
    }

    ## Checking everything
    # 保存或更新 `params` 的值。
    params = check_quad_param_limits(params=params)
    # 返回当前函数的结果。
    return params

# 定义函数 `sample_nodelay`。
def sample_nodelay(params):
    # 保存或更新 `params[motor][damp_time_up]` 的值。
    params["motor"]["damp_time_up"] = 0.
    # 保存或更新 `params[motor][damp_time_down]` 的值。
    params["motor"]["damp_time_down"] = 0.
    # 返回当前函数的结果。
    return params

# 定义函数 `sample_linearity`。
def sample_linearity(params):
    # 保存或更新 `params[motor][linearity]` 的值。
    params["motor"]["linearity"] = np.random.uniform(low=0., high=1.)
    # 返回当前函数的结果。
    return params

# 定义函数 `sample_t2w`。
def sample_t2w(params, t2w_min, t2w_max):
    # 保存或更新 `params[motor][thrust_to_weight]` 的值。
    params["motor"]["thrust_to_weight"] = np.random.uniform(low=t2w_min, high=t2w_max)
    # 返回当前函数的结果。
    return params

# 定义函数 `sample_t2w_t2t`。
def sample_t2w_t2t(params, t2w_min, t2w_max, t2t_min=0.003, t2t_max=0.009):
    # 保存或更新 `params[motor][thrust_to_weight]` 的值。
    params["motor"]["thrust_to_weight"] = np.random.uniform(low=t2w_min, high=t2w_max)
    # 保存或更新 `params[motor][torque_to_thrust]` 的值。
    params["motor"]["torque_to_thrust"] = np.random.uniform(low=t2t_min, high=t2t_max)
    # 返回当前函数的结果。
    return params

# 定义函数 `sample_simplified_random_dyn`。
def sample_simplified_random_dyn():
    # 下面开始文档字符串说明。
    """
    The function samples parameters for all possible quadrotors
    Args:
        scale (float): scale of sampling
    Returns:
        dict: sampled quadrotor parameters
    """
    ###################################################################
    ## Masses and dimensions
    # Estimated Crazyflie mass / dimension / arm length : ~0.027 [kg] / 0.065 x 0.065 [m] / 0.092 [m]
    # Estimated Hummingbird mass / dimension /arm length : ~0.547 [kg] / 0.764 x 0.764 [m] / 0.540 [m]
    # 保存或更新 `geom_params` 的值。
    geom_params = {}

    # 保存或更新 `geom_params[mass]` 的值。
    geom_params["mass"] = np.random.uniform(low=0.020, high=0.6)
    ###################################################################
    ## GEOMETRIES
    ## arm length here represents the diagonal motor to motor distance
    # 保存或更新 `arm_length` 的值。
    arm_length = np.random.uniform(low=0.05, high=0.5)
    # 保存或更新 `geom_params[arms]` 的值。
    geom_params["arms"] = {"l": arm_length}
    # 保存或更新 `motor_pos_x` 的值。
    motor_pos_x = motor_pos_y = arm_length * np.sqrt(2) / 4
    # 保存或更新 `geom_params[motor_pos]` 的值。
    geom_params["motor_pos"] = {"xyz": [motor_pos_x, motor_pos_y, 0.0]}

    # 保存或更新 `thrust_to_weight` 的值。
    thrust_to_weight = np.random.uniform(low=1.5, high=3.5)

    ## Damping parameters
    # damp_vel_scale = np.random.uniform(low=0.01, high=2.)
    # damp_omega_scale = damp_vel_scale * np.random.uniform(low=0.75, high=1.25)
    # damp_params = {
    #     "vel": 0.001 * damp_vel_scale, 
    #     "omega_quadratic": 0.015 * damp_omega_scale}
    # 保存或更新 `damp_params` 的值。
    damp_params = {
        "vel": 0.0, 
        "omega_quadratic": 0.0}

    ## Noise parameters
    # 保存或更新 `noise_params` 的值。
    noise_params = {}
    # 保存或更新 `noise_params[thrust_noise_ratio]` 的值。
    noise_params["thrust_noise_ratio"] = np.random.uniform(low=0.05, high=0.1) #0.01
    
    ## Motor parameters
    # 保存或更新 `damp_time_up` 的值。
    damp_time_up = np.random.uniform(low=0.1, high=0.2)
    # 保存或更新 `damp_time_down_scale` 的值。
    damp_time_down_scale = np.random.uniform(low=1.0, high=2.0)
    # 保存或更新 `motor_params` 的值。
    motor_params = {"thrust_to_weight" : thrust_to_weight,
                    "torque_to_thrust": np.random.uniform(low=0.005, high=0.02), #0.05 originally
                    "assymetry": np.random.uniform(low=0.9, high=1.1, size=4),
                    "linearity": 1.0,
                    "C_drag": 0.,
                    "C_roll": 0.,
                    "damp_time_up": damp_time_up,
                    "damp_time_down": damp_time_down_scale * damp_time_up
                    # "linearity": np.random.normal(loc=0.5, scale=0.1)
                    }

    ## Summarizing
    # 保存或更新 `params` 的值。
    params = {
        "geom": geom_params, 
        "damp": damp_params, 
        "noise": noise_params,
        "motor": motor_params
    }

    ## Checking everything
    # params = check_quad_param_limits(params=params)
    # 返回当前函数的结果。
    return params


# 定义类 `Crazyflie`。
class Crazyflie(object):
    # 定义函数 `sample`。
    def sample(self, params=None):
        # 返回当前函数的结果。
        return crazyflie_params()

# 定义类 `DefaultQuad`。
class DefaultQuad(object):
    # 定义函数 `sample`。
    def sample(self, params=None):
        # 返回当前函数的结果。
        return defaultquad_params()

# 定义类 `MediumQuad`。
class MediumQuad(object):
    # 定义函数 `sample`。
    def sample(self, params=None):
        # 返回当前函数的结果。
        return mediumquad_params()

# 定义类 `RandomQuad`。
class RandomQuad(object):
    # 定义函数 `sample`。
    def sample(self, params=None):
        # 返回当前函数的结果。
        return randomquad_parameters()

# 定义类 `RelativeSampler`。
class RelativeSampler(object):
    # 定义函数 `__init__`。
    def __init__(self, params, noise_ratio=0., noise_ratio_custom=None, sampler="normal"):
        # 保存或更新 `noise_params` 的值。
        self.noise_params = get_dyn_randomization_params(
                        params, 
                        noise_ratio=noise_ratio, 
                        noise_ratio_params=noise_ratio_custom)
        # 保存或更新 `sampler` 的值。
        self.sampler = sampler
    # 定义函数 `sample`。
    def sample(self, params):
        # 返回当前函数的结果。
        return perturb_dyn_parameters(
            params=params, 
            noise_params=self.noise_params, 
            sampler=self.sampler
        )

# 定义类 `AbsoluteSampler`。
class AbsoluteSampler(object):
    # 定义函数 `__init__`。
    def __init__(self, params, noise_params, sampler="uniform"):
        # 保存或更新 `noise_params` 的值。
        self.noise_params = copy.deepcopy(noise_params)
        # 保存或更新 `sampler` 的值。
        self.sampler = sampler
        
    # 定义函数 `sample`。
    def sample(self, params):
        # 返回当前函数的结果。
        return resample_dyn_parameters(
            params=params, 
            noise_params=self.noise_params, 
            sampler=self.sampler
        )

# 定义类 `ConstValueSampler`。
class ConstValueSampler(object):
    # 定义函数 `__init__`。
    def __init__(self, params, params_change):
        # 保存或更新 `params_change` 的值。
        self.params_change = copy.deepcopy(params_change)
        
    # 定义函数 `sample`。
    def sample(self, params):
        # 保存或更新 `dict_update_existing(params, dic_upd` 的值。
        dict_update_existing(params, dic_upd=self.params_change)
        # 返回当前函数的结果。
        return params

    # def sample_random_nondim_dyn():
    #     """
    #     The function samples parameters for all possible non-dimensional quadrotors
    #     Args:
    #         scale (float): scale of sampling
    #     Returns:
    #         dict: sampled quadrotor parameters
    #     """
    #     ###################################################################
    #     ## DENSITIES (body, payload, arms, motors, propellers)
    #     # Crazyflie estimated body / payload / arms / motors / props density: 1388.9 / 1785.7 / 1777.8 / 1948.8 / 246.6 kg/m^3
    #     # Hummingbird estimated body / payload / arms / motors/ props density: 588.2 / 173.6 / 1111.1 / 509.3 / 246.6 kg/m^3
    #     geom_params = {}
       
    #     geom_params["body"] = {"mass": 1.0}
    #     geom_params["payload"] = {"mass": 0}
    #     geom_params["arms"]    = {"mass": 0.}
    #     geom_params["motors"]  = {"mass": 0.}
    #     geom_params["propellers"] = {"mass": 0.}

    #     ###################################################################
    #     ## GEOMETRIES
    #     # MOTORS (and overal size)
    #     roll_authority = np.random.uniform(low=600, high=1200) #for our current low inertia CF ~ 1050
    #     pitch_authority = np.random.uniform(low=0.8, high=1.0) * roll_authority
    #     total_w = np.random.uniform(low=0.5, high=0.5)
    #     total_l = total_w
    #     motor_z = np.random.normal(loc=0., scale=total_w / 8.)
    #     geom_params["motor_pos"] = {"xyz": [total_w / 2., total_l / 2., motor_z]}
    #     geom_params["motors"]["r"] = total_w * np.random.normal(loc=0.1, scale=0.01)
    #     geom_params["motors"]["h"] = geom_params["motors"]["r"] * np.random.normal(loc=1.0, scale=0.05)

    #     # BODY
    #     geom_params["body"]["w"] = np.random.uniform(low=1.0, high=1.0)
    #     ## Promotes more elangeted bodies when they are more narrow
    #     geom_params["body"]["l"] =  np.random.uniform(low=1.0, high=2.0) * geom_params["body"]["w"]
    #     geom_params["body"]["h"] =  np.random.uniform(low=0.1, high=1.0) * geom_params["body"]["w"]
        


    #     # PAYLOAD
    #     pl_scl = np.random.uniform(low=0.25, high=1.0, size=3)
    #     geom_params["payload"]["w"] =  pl_scl[0] * geom_params["body"]["w"]
    #     geom_params["payload"]["l"] =  pl_scl[1] * geom_params["body"]["l"]
    #     geom_params["payload"]["h"] =  pl_scl[2] * geom_params["body"]["h"]
    #     geom_params["payload_pos"] = {
    #             "xy": np.random.normal(loc=0., scale=geom_params["body"]["w"] / 10., size=2), 
    #             "z_sign": np.sign(np.random.uniform(low=-1, high=1))}
    #     # z_sing corresponds to location (+1 - on top of the body, -1 - on the bottom of the body)

    #     # ARMS
    #     geom_params["arms"]["w"] = total_w * np.random.normal(loc=0.05, scale=0.005)
    #     geom_params["arms"]["h"] = total_w * np.random.normal(loc=0.05, scale=0.005)
    #     geom_params["arms_pos"] = {"angle": np.random.normal(loc=45., scale=10.), "z": motor_z - geom_params["motors"]["h"]/2.}
        
    #     # PROPS
    #     thrust_to_weight = np.random.uniform(low=1.8, high=2.5)
    #     geom_params["propellers"]["h"] = 0.01
    #     geom_params["propellers"]["r"] = (0.3) * total_w * (thrust_to_weight / 2.0)**0.5
        
    #     ## Damping parameters
    #     # damp_vel_scale = np.random.uniform(low=0.01, high=2.)
    #     # damp_omega_scale = damp_vel_scale * np.random.uniform(low=0.75, high=1.25)
    #     # damp_params = {
    #     #     "vel": 0.001 * damp_vel_scale, 
    #     #     "omega_quadratic": 0.015 * damp_omega_scale}
    #     damp_params = {
    #         "vel": 0.0, 
    #         "omega_quadratic": 0.0}

    #     ## Noise parameters
    #     noise_params = {}
    #     noise_params["thrust_noise_ratio"] = np.random.uniform(low=0.01, high=0.05) #0.01
        
    #     ## Motor parameters
    #     damp_time_up = np.random.uniform(low=0.1, high=0.2)
    #     damp_time_down_scale = np.random.uniform(low=1.0, high=2.0)
    #     motor_params = {"thrust_to_weight" : thrust_to_weight,
    #                     "torque_to_thrust": np.random.uniform(low=0.005, high=0.025), #0.05 originally
    #                     "assymetry": np.random.uniform(low=0.9, high=1.1, size=4),
    #                     "linearity": 1.0,
    #                     "C_drag": 0.,
    #                     "C_roll": 0.,
    #                     "damp_time_up": damp_time_up,
    #                     "damp_time_down": damp_time_down_scale * damp_time_up
    #                     # "linearity": np.random.normal(loc=0.5, scale=0.1)
    #                     }

    #     ## Summarizing
    #     params = {
    #         "geom": geom_params, 
    #         "damp": damp_params, 
    #         "noise": noise_params,
    #         "motor": motor_params
    #     }

    #     ## Checking everything
    #     params = check_quad_param_limits(params=params)
    #     return params
