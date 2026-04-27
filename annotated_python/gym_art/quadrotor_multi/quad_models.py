# 中文注释副本；原始文件：gym_art/quadrotor_multi/quad_models.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。
# 这个文件集中给出几套标准四旋翼机体参数模板。
# 上游调用者通常是动力学初始化、惯量计算和 domain randomization；下游则是 `QuadLink`/`QuadLinkSimplified`
# 这类装配器，以及真正运行仿真的 `quadrotor_dynamics.py`。


def crazyflie_params():
    # 这组模板对应论文和仓库里最常见的小型 Crazyflie 风格机体。
    # 重点不是单个数字本身，而是把几何尺寸、阻尼、噪声和电机模型拆成统一的四段字典，方便后续模块按键读取。
    geom_params = {}
    geom_params["body"] = {"l": 0.03, "w": 0.03, "h": 0.004, "m": 0.005}
    geom_params["payload"] = {"l": 0.035, "w": 0.02, "h": 0.008, "m": 0.01}
    geom_params["arms"] = {"l": 0.022, "w": 0.005, "h": 0.005, "m": 0.001}
    geom_params["motors"] = {"h": 0.02, "r": 0.0035, "m": 0.0015}
    geom_params["propellers"] = {"h": 0.002, "r": 0.022, "m": 0.00075}

    geom_params["motor_pos"] = {"xyz": [0.065 / 2, 0.065 / 2, 0.]}
    geom_params["arms_pos"] = {"angle": 45., "z": 0.}
    geom_params["payload_pos"] = {"xy": [0., 0.], "z_sign": 1}

    damp_params = {"vel": 0.0, "omega_quadratic": 0.0}

    noise_params = {}
    noise_params["thrust_noise_ratio"] = 0.05

    motor_params = {
        "thrust_to_weight": 1.9,
        "assymetry": [1.0, 1.0, 1.0, 1.0],
        "torque_to_thrust": 0.006,
        "linearity": 1.0,
        "C_drag": 0.000,
        "C_roll": 0.000,
        "damp_time_up": 0.15,
        "damp_time_down": 0.15
    }

    params = {
        "geom": geom_params,
        "damp": damp_params,
        "noise": noise_params,
        "motor": motor_params
    }
    return params


def defaultquad_params():
    # 这组更接近较大的 Hummingbird/通用研究用 quad 模板，
    # 常用来和 Crazyflie 尺度做对照，或作为默认较大机体的动力学基线。
    geom_params = {}
    geom_params["body"] = {"l": 0.1, "w": 0.1, "h": 0.085, "m": 0.5}
    geom_params["payload"] = {"l": 0.12, "w": 0.12, "h": 0.04, "m": 0.1}
    geom_params["arms"] = {"l": 0.1, "w": 0.015, "h": 0.015, "m": 0.025}
    geom_params["motors"] = {"h": 0.02, "r": 0.025, "m": 0.02}
    geom_params["propellers"] = {"h": 0.001, "r": 0.1, "m": 0.009}

    geom_params["motor_pos"] = {"xyz": [0.12, 0.12, 0.]}
    geom_params["arms_pos"] = {"angle": 45., "z": 0.}
    geom_params["payload_pos"] = {"xy": [0., 0.], "z_sign": -1}

    damp_params = {"vel": 0.0, "omega_quadratic": 0.0}

    noise_params = {}
    noise_params["thrust_noise_ratio"] = 0.05

    motor_params = {
        "thrust_to_weight": 2.8,
        "assymetry": [1.0, 1.0, 1.0, 1.0],
        "torque_to_thrust": 0.05,
        "linearity": 1.0,
        "C_drag": 0.,
        "C_roll": 0.,
        "damp_time_up": 0,
        "damp_time_down": 0
    }

    params = {
        "geom": geom_params,
        "damp": damp_params,
        "noise": noise_params,
        "motor": motor_params
    }
    return params


def mediumquad_params():
    # 这一组介于小型 Crazyflie 和较大默认机体之间，常用于“中等尺度”实验。
    # 其结构和其它模板保持一致，这样上层切换机体尺寸时不需要改配置消费逻辑。
    geom_params = {}
    geom_params["body"] = {"l": 0.04, "w": 0.04, "h": 0.04, "m": 0.04}
    geom_params["payload"] = {"l": 0.06, "w": 0.015, "h": 0.015, "m": 0.029}
    geom_params["arms"] = {"l": 0.04, "w": 0.01, "h": 0.003, "m": 0.006}
    geom_params["motors"] = {"h": 0.013, "r": 0.007, "m": 0.006}
    geom_params["propellers"] = {"h": 0.007, "r": 0.035, "m": 0.0012}

    geom_params["motor_pos"] = {"xyz": [0.046, 0.046, 0.]}
    geom_params["arms_pos"] = {"angle": 45., "z": 0.}
    geom_params["payload_pos"] = {"xy": [0., 0.], "z_sign": -1}

    damp_params = {"vel": 0.0, "omega_quadratic": 0.0}

    noise_params = {}
    noise_params["thrust_noise_ratio"] = 0.05

    motor_params = {
        "thrust_to_weight": 2.5,
        "assymetry": [1.0, 1.0, 1.0, 1.0],
        "torque_to_thrust": 0.05,
        "linearity": 1.0,
        "C_drag": 0.,
        "C_roll": 0.,
        "damp_time_up": 0.15,
        "damp_time_down": 0.15
    }

    params = {
        "geom": geom_params,
        "damp": damp_params,
        "noise": noise_params,
        "motor": motor_params
    }
    return params


def crazyflie_lowinertia_params():
    # 这组模板专门把 Crazyflie 风格机体的若干部件减重，
    # 用来构造“外形近似不变但惯量显著更低”的对照机体，方便测试动力学灵敏度。
    geom_params = {}
    geom_params["body"] = {"l": 0.03, "w": 0.03, "h": 0.004, "m": 0.014}
    geom_params["payload"] = {"l": 0.035, "w": 0.02, "h": 0.008, "m": 0.01}
    geom_params["arms"] = {"l": 0.022, "w": 0.005, "h": 0.005, "m": 0.0005}
    geom_params["motors"] = {"h": 0.02, "r": 0.0035, "m": 0.0005}
    geom_params["propellers"] = {"h": 0.002, "r": 0.022, "m": 0.0000075}

    geom_params["motor_pos"] = {"xyz": [0.065 / 2, 0.065 / 2, 0.]}
    geom_params["arms_pos"] = {"angle": 45., "z": 0.}
    geom_params["payload_pos"] = {"xy": [0., 0.], "z_sign": 1}

    damp_params = {"vel": 0.0, "omega_quadratic": 0.0}

    noise_params = {}
    noise_params["thrust_noise_ratio"] = 0.05

    motor_params = {
        "thrust_to_weight": 1.9,
        "assymetry": [1.0, 1.0, 1.0, 1.0],
        "torque_to_thrust": 0.006,
        "linearity": 1.0,
        "C_drag": 0.000,
        "C_roll": 0.000,
        "damp_time_up": 0.15,
        "damp_time_down": 0.15
    }

    params = {
        "geom": geom_params,
        "damp": damp_params,
        "noise": noise_params,
        "motor": motor_params
    }
    return params
