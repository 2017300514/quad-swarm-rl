# 中文注释副本；原始文件：swarm_rl/sim2real/sim2real.py
# 说明：为避免修改源码，本文件仅作为阅读辅助材料。

# 导入当前模块依赖。
import argparse
import json
import os
from distutils.util import strtobool
from pathlib import Path

# 导入当前模块依赖。
import torch
import torch.nn as nn
from attrdict import AttrDict
from typing import List

# 导入当前模块依赖。
from sample_factory.model.actor_critic import create_actor_critic

# 导入当前模块依赖。
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
from swarm_rl.sim2real.code_blocks import (
    headers_network_evaluate,
    headers_evaluation,
    linear_activation,
    sigmoid_activation,
    relu_activation,
    single_drone_eval,
    multi_drone_attn_eval,
    headers_multi_agent_attention,
    attention_body
)
# 导入当前模块依赖。
from swarm_rl.train import register_swarm_components


# 定义函数 `parse_args`。
def parse_args():
    # 保存或更新 `parser` 的值。
    parser = argparse.ArgumentParser()
    # 保存或更新 `parser.add_argument(--torch_model_dir, type` 的值。
    parser.add_argument('--torch_model_dir', type=str, default='swarm_rl/sim2real/torch_models/single',
                        help='Path where the policy and cfg is stored')
    # 保存或更新 `parser.add_argument(--output_dir, type` 的值。
    parser.add_argument('--output_dir', type=str, default='swarm_rl/sim2real/c_models',
                        help='Where you want the c model to be saved')
    # 保存或更新 `parser.add_argument(--output_model_name, type` 的值。
    parser.add_argument('--output_model_name', type=str, default='model.c')
    # 保存或更新 `parser.add_argument(--testing, type` 的值。
    parser.add_argument('--testing', type=lambda x: bool(strtobool(x)), default=False,
                        help='Whether or not to save the c model in testing mode. Enable this if you want to run the '
                             'unit test to make sure the output of the c model is the same as the pytorch model. Set '
                             'to False if you want to output a c model that will be actually used for sim2real')
    # 保存或更新 `parser.add_argument(--model_type, type` 的值。
    parser.add_argument('--model_type', type=str, choices=['single', 'attention'],
                        help='What kind of model we are working with. '
                             'Currently only single drone models are supported.')
    # 保存或更新 `args` 的值。
    args = parser.parse_args()
    # 返回当前函数的结果。
    return AttrDict(vars(args))


# 定义函数 `torch_to_c_model`。
def torch_to_c_model(args):
    # 保存或更新 `model_dir` 的值。
    model_dir = Path(args.torch_model_dir)
    # 保存或更新 `model` 的值。
    model = load_sf_model(model_dir, args.model_type)

    # 保存或更新 `output_dir` 的值。
    output_dir = Path(args.output_dir)
    # 保存或更新 `output_path` 的值。
    output_path = output_dir.joinpath(args.model_type, args.output_model_name)
    # 保存或更新 `output_folder` 的值。
    output_folder = output_dir.joinpath(args.model_type)
    # 根据条件决定是否进入当前分支。
    if args.model_type == 'single':
        # 保存或更新 `generate_c_model(model, str(output_path), str(output_folder), testing` 的值。
        generate_c_model(model, str(output_path), str(output_folder), testing=args.testing)
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 保存或更新 `generate_c_model_attention(model, str(output_path), str(output_folder), testing` 的值。
        generate_c_model_attention(model, str(output_path), str(output_folder), testing=args.testing)


# 定义函数 `load_sf_model`。
def load_sf_model(model_dir: Path, model_type: str):
    # 下面开始文档字符串说明。
    """
        Load a trained SF pytorch model
    """
    # 断言当前条件成立，用于保护运行假设。
    assert model_dir.exists(), f'Path {str(model_dir)} is not a valid path'
    # Load hyper-parameters
    # 保存或更新 `cfg_path` 的值。
    cfg_path = model_dir.joinpath('config.json')
    # 使用上下文管理器包裹后续资源操作。
    with open(cfg_path, 'r') as f:
        # 保存或更新 `args` 的值。
        args = json.load(f)
    # 保存或更新 `args` 的值。
    args = AttrDict(args)

    # Manually set some values
    # 保存或更新 `args.visualize_v_value` 的值。
    args.visualize_v_value = False
    # 执行这一行逻辑。
    args.quads_encoder_type = 'attention' if model_type == 'attention' else 'corl'
    # 保存或更新 `args.quads_obstacle_scan_range` 的值。
    args.quads_obstacle_scan_range = 0
    # 保存或更新 `args.quads_obstacle_ray_num` 的值。
    args.quads_obstacle_ray_num = 0
    # 保存或更新 `args.quads_sim2real` 的值。
    args.quads_sim2real = True
    # 保存或更新 `args.quads_domain_random` 的值。
    args.quads_domain_random = False
    # 保存或更新 `args.quads_obst_density_random` 的值。
    args.quads_obst_density_random = False
    # 保存或更新 `args.quads_obst_density_min` 的值。
    args.quads_obst_density_min = 0
    # 保存或更新 `args.quads_obst_density_max` 的值。
    args.quads_obst_density_max = 0
    # 保存或更新 `args.quads_obst_size_random` 的值。
    args.quads_obst_size_random = False
    # 保存或更新 `args.quads_obst_size_min` 的值。
    args.quads_obst_size_min = 0
    # 保存或更新 `args.quads_obst_size_max` 的值。
    args.quads_obst_size_max = 0

    # Load model
    # 调用 `register_swarm_components` 执行当前处理。
    register_swarm_components()
    # spawn a dummy env, so we can get the obs and action space info
    # 保存或更新 `env` 的值。
    env = make_quadrotor_env_multi(args)
    # 保存或更新 `model` 的值。
    model = create_actor_critic(args, env.observation_space, env.action_space)
    # 保存或更新 `model_path` 的值。
    model_path = list(model_dir.glob('*.pth'))[0]
    # 调用 `load_state_dict` 执行当前处理。
    model.load_state_dict(torch.load(model_path)['model'])

    # 返回当前函数的结果。
    return model


# 定义函数 `process_layer`。
def process_layer(name: str, param: nn.Parameter, type: str):
    # 下面开始文档字符串说明。
    '''
    Convert a torch parameter from the NN into a c-equivalent represented as a string 
    '''
    # 根据条件决定是否进入当前分支。
    if type == 'weight':
        # 保存或更新 `weight` 的值。
        weight = 'static const float ' + name + '[' + str(param.shape[0]) + '][' + str(param.shape[1]) + '] = {'
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for row in param:
            weight += '{'
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for num in row:
                weight += str(num.item()) + ','
            # get rid of comma after the last number
            weight = weight[:-1]
            weight += '},'
        # get rid of comma after the last curly bracket
        weight = weight[:-1]
        weight += '};\n'
        # 返回当前函数的结果。
        return weight
    # 当前置条件都不满足时，执行兜底分支。
    else:
        # 保存或更新 `bias` 的值。
        bias = 'static const float ' + name + '[' + str(param.shape[0]) + '] = {'
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for num in param:
            bias += str(num.item()) + ','
        # get rid of comma after last number
        bias = bias[:-1]
        bias += '};\n'
        # 返回当前函数的结果。
        return bias


# 定义函数 `generate_c_weights_attention`。
def generate_c_weights_attention(model: nn.Module, transpose: bool = False):
    # 下面开始文档字符串说明。
    """
            Generate c friendly weight strings for the c version of the attention model
            order is: self-encoder, neighbor-encoder, obst-encoder, attention, then final combined output layers 
    """
    # 同时更新 `self_weights`, `self_biases`, `self_layer_names`, `self_bias_names` 等变量。
    self_weights, self_biases, self_layer_names, self_bias_names = [], [], [], []
    # 同时更新 `neighbor_weights`, `neighbor_biases`, `nbr_layer_names`, `nbr_bias_names` 等变量。
    neighbor_weights, neighbor_biases, nbr_layer_names, nbr_bias_names = [], [], [], []
    # 同时更新 `obst_weights`, `obst_biases`, `obst_layer_names`, `obst_bias_names` 等变量。
    obst_weights, obst_biases, obst_layer_names, obst_bias_names = [], [], [], []
    # 同时更新 `attn_weights`, `attn_biases`, `attn_layer_names`, `attn_bias_names` 等变量。
    attn_weights, attn_biases, attn_layer_names, attn_bias_names = [], [], [], []
    # 同时更新 `out_weights`, `out_biases`, `out_layer_names`, `out_bias_names` 等变量。
    out_weights, out_biases, out_layer_names, out_bias_names = [], [], [], [],
    # 保存或更新 `outputs` 的值。
    outputs = []
    # 同时更新 `n_self`, `n_nbr`, `n_obst` 等变量。
    n_self, n_nbr, n_obst = 0, 0, 0
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for name, param in model.named_parameters():
        # get the self encoder weights 
        # 根据条件决定是否进入当前分支。
        if transpose:
            # 保存或更新 `param` 的值。
            param = param.T
        # 保存或更新 `c_name` 的值。
        c_name = name.replace('.', '_')
        # 根据条件决定是否进入当前分支。
        if 'weight' in c_name and 'critic' not in c_name and 'layer_norm' not in c_name:
            # 保存或更新 `weight` 的值。
            weight = process_layer(c_name, param, type='weight')
            # 根据条件决定是否进入当前分支。
            if 'self_embed' in c_name:
                # 调用 `append` 执行当前处理。
                self_layer_names.append(name)
                # 调用 `append` 执行当前处理。
                self_weights.append(weight)
                # 调用 `append` 执行当前处理。
                outputs.append('static float output_' + str(n_self) + '[' + str(param.shape[1]) + '];\n')
                # 保存或更新 `n_self` 的值。
                n_self += 1
            # 当上一分支不满足时，继续判断新的条件。
            elif 'neighbor_embed' in c_name:
                # 调用 `append` 执行当前处理。
                nbr_layer_names.append(name)
                # 调用 `append` 执行当前处理。
                neighbor_weights.append(weight)
                # 调用 `append` 执行当前处理。
                outputs.append('static float nbr_output_' + str(n_nbr) + '[' + str(param.shape[1]) + '];\n')
                # 保存或更新 `n_nbr` 的值。
                n_nbr += 1
            # 当上一分支不满足时，继续判断新的条件。
            elif 'obstacle_embed' in c_name:
                # 调用 `append` 执行当前处理。
                obst_layer_names.append(name)
                # 调用 `append` 执行当前处理。
                obst_weights.append(weight)
                # 调用 `append` 执行当前处理。
                outputs.append('static float obst_output_' + str(n_obst) + '[' + str(param.shape[1]) + '];\n')
                # 保存或更新 `n_obst` 的值。
                n_obst += 1
            # 当上一分支不满足时，继续判断新的条件。
            elif 'attention' in c_name or 'layer_norm' in c_name:
                # 调用 `append` 执行当前处理。
                attn_layer_names.append(name)
                # 调用 `append` 执行当前处理。
                attn_weights.append(weight)
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # output layer
                # 调用 `append` 执行当前处理。
                out_layer_names.append(name)
                # 调用 `append` 执行当前处理。
                out_weights.append(weight)
                # these will be considered part of the self encoder
                # 调用 `append` 执行当前处理。
                outputs.append('static float output_' + str(n_self) + '[' + str(param.shape[1]) + '];\n')
                # 保存或更新 `n_self` 的值。
                n_self += 1
        # 根据条件决定是否进入当前分支。
        if ('bias' in c_name or 'layer_norm' in c_name) and 'critic' not in c_name:
            # 保存或更新 `bias` 的值。
            bias = process_layer(c_name, param, type='bias')
            # 根据条件决定是否进入当前分支。
            if 'self_embed' in c_name:
                # 调用 `append` 执行当前处理。
                self_bias_names.append(name)
                # 调用 `append` 执行当前处理。
                self_biases.append(bias)
            # 当上一分支不满足时，继续判断新的条件。
            elif 'neighbor_embed' in c_name:
                # 调用 `append` 执行当前处理。
                nbr_bias_names.append(name)
                # 调用 `append` 执行当前处理。
                neighbor_biases.append(bias)
            # 当上一分支不满足时，继续判断新的条件。
            elif 'obstacle_embed' in c_name:
                # 调用 `append` 执行当前处理。
                obst_bias_names.append(name)
                # 调用 `append` 执行当前处理。
                obst_biases.append(bias)
            # 当上一分支不满足时，继续判断新的条件。
            elif 'attention' in c_name or 'layer_norm' in c_name:
                # 调用 `append` 执行当前处理。
                attn_bias_names.append(name)
                # 调用 `append` 执行当前处理。
                attn_biases.append(bias)
            # 当前置条件都不满足时，执行兜底分支。
            else:
                # output layer
                # 调用 `append` 执行当前处理。
                out_bias_names.append(name)
                # 调用 `append` 执行当前处理。
                out_biases.append(bias)

    # 保存或更新 `self_layer_names` 的值。
    self_layer_names += out_layer_names
    # 保存或更新 `self_bias_names` 的值。
    self_bias_names += out_bias_names
    # 保存或更新 `self_weights` 的值。
    self_weights += out_weights
    # 保存或更新 `self_biases` 的值。
    self_biases += out_biases
    # 保存或更新 `info` 的值。
    info = {
        'encoders': {
            'self': [self_layer_names, self_bias_names, self_weights, self_biases],
            'nbr': [nbr_layer_names, nbr_bias_names, neighbor_weights, neighbor_biases],
            'obst': [obst_layer_names, obst_bias_names, obst_weights, obst_biases],
            'attn': [attn_layer_names, attn_bias_names, attn_weights, attn_biases],
        },
        'out': [out_layer_names, out_bias_names, out_weights, out_biases],
        'outputs': outputs
    }

    # 返回当前函数的结果。
    return info


# 定义函数 `generate_c_weights`。
def generate_c_weights(model: nn.Module, transpose: bool = False):
    # 下面开始文档字符串说明。
    """
        Generate c friendly weight strings for the c version of the model
    """
    # 同时更新 `weights`, `biases` 等变量。
    weights, biases = [], []
    # 同时更新 `layer_names`, `bias_names`, `outputs` 等变量。
    layer_names, bias_names, outputs = [], [], []
    # 保存或更新 `n_bias` 的值。
    n_bias = 0
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for name, param in model.named_parameters():
        # 根据条件决定是否进入当前分支。
        if transpose:
            # 保存或更新 `param` 的值。
            param = param.T
        # 保存或更新 `name` 的值。
        name = name.replace('.', '_')
        # 根据条件决定是否进入当前分支。
        if 'weight' in name and 'critic' not in name and 'layer_norm' not in name:
            # 调用 `append` 执行当前处理。
            layer_names.append(name)
            # 保存或更新 `weight` 的值。
            weight = 'static const float ' + name + '[' + str(param.shape[0]) + '][' + str(param.shape[1]) + '] = {'
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for row in param:
                weight += '{'
                # 遍历当前序列或迭代器，逐项执行下面的逻辑。
                for num in row:
                    weight += str(num.item()) + ','
                # get rid of comma after the last number
                weight = weight[:-1]
                weight += '},'
            # get rid of comma after the last curly bracket
            weight = weight[:-1]
            weight += '};\n'
            # 调用 `append` 执行当前处理。
            weights.append(weight)

        # 根据条件决定是否进入当前分支。
        if 'bias' in name or 'layer_norm' in name and 'critic' not in name:
            # 调用 `append` 执行当前处理。
            bias_names.append(name)
            # 保存或更新 `bias` 的值。
            bias = 'static const float ' + name + '[' + str(param.shape[0]) + '] = {'
            # 遍历当前序列或迭代器，逐项执行下面的逻辑。
            for num in param:
                bias += str(num.item()) + ','
            # get rid of comma after last number
            bias = bias[:-1]
            bias += '};\n'
            # 调用 `append` 执行当前处理。
            biases.append(bias)
            # 保存或更新 `output` 的值。
            output = 'static float output_' + str(n_bias) + '[' + str(param.shape[0]) + '];\n'
            # 调用 `append` 执行当前处理。
            outputs.append(output)
            # 保存或更新 `n_bias` 的值。
            n_bias += 1

    # 返回当前函数的结果。
    return layer_names, bias_names, weights, biases, outputs


# 定义函数 `self_encoder_c_str`。
def self_encoder_c_str(prefix: str, weight_names: List[str], bias_names: List[str]):
    # 保存或更新 `method` 的值。
    method = """void networkEvaluate(struct control_t_n *control_n, const float *state_array) {"""
    num_layers = len(weight_names)
    # write the for loops for forward-prop
    for_loops = []
    input_for_loop = f'''
        for (int i = 0; i < {prefix}_structure[0][1]; i++) {{
            output_0[i] = 0;
            for (int j = 0; j < {prefix}_structure[0][0]; j++) {{
                output_0[i] += state_array[j] * {weight_names[0].replace('.', '_')}[j][i];
            }}
            output_0[i] += {bias_names[0].replace('.', '_')}[i];
            output_0[i] = tanhf(output_0[i]);
        }}
    '''
    for_loops.append(input_for_loop)

    # rest of the hidden layers
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for n in range(1, num_layers - 1):
        for_loop = f'''
            for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
                output_{str(n)}[i] = 0;
                for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                    output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
                }}
                output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
                output_{str(n)}[i] = tanhf(output_{str(n)}[i]);
            }}
        '''
        for_loops.append(for_loop)

    # the last hidden layer which is supposed to have no non-linearity
    n = num_layers - 1
    output_for_loop = f'''
                for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
                    output_{str(n)}[i] = 0;
                    for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                        output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
                    }}
                    output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
                }}
    '''
    for_loops.append(output_for_loop)

    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for code in for_loops:
        method += code
    # 根据条件决定是否进入当前分支。
    if 'self' in prefix:
        # assign network outputs to control
        assignment = """
                    control_n->thrust_0 = output_""" + str(n) + """[0];
                    control_n->thrust_1 = output_""" + str(n) + """[1];
                    control_n->thrust_2 = output_""" + str(n) + """[2];
                    control_n->thrust_3 = output_""" + str(n) + """[3];	
            """
        method += assignment
    # closing bracket
    method += """}\n\n"""
    # 返回当前函数的结果。
    return method


# 定义函数 `self_encoder_attn_c_str`。
def self_encoder_attn_c_str(prefix: str, weight_names: List[str], bias_names: List[str]):
    # 保存或更新 `method` 的值。
    method = """void networkEvaluate(struct control_t_n *control_n, const float *state_array) {"""
    num_layers = len(weight_names)
    # write the for loops for forward-prop of self embed layer
    for_loops = []
    input_for_loop = f'''
        // Self embed layer
        for (int i = 0; i < {prefix}_structure[0][1]; i++) {{
            output_0[i] = 0;
            for (int j = 0; j < {prefix}_structure[0][0]; j++) {{
                output_0[i] += state_array[j] * {weight_names[0].replace('.', '_')}[j][i];
            }}
            output_0[i] += {bias_names[0].replace('.', '_')}[i];
            output_0[i] = tanhf(output_0[i]);
        }}
    '''
    for_loops.append(input_for_loop)

    # concat self embedding and attention embedding
    # for n in range(1, num_layers - 1):
    for_loop = f'''
        // Concat self_embed, neighbor_embed and obst_embed
        for (int i = 0; i < self_structure[0][1]; i++) {{
            output_embeds[i] = output_0[i];
            output_embeds[i + self_structure[0][1]] = attn_embeds[0][i];
            output_embeds[i + 2 * self_structure[0][1]] = attn_embeds[1][i];
        }}
    '''
    for_loops.append(for_loop)

    # forward-prop of feedforward layer
    output_for_loop = f'''
        // Feedforward layer
        for (int i = 0; i < self_structure[1][1]; i++) {{
            output_1[i] = 0;
            for (int j = 0; j < 3 * self_structure[0][1]; j++) {{
                output_1[i] += output_embeds[j] * actor_encoder_feed_forward_0_weight[j][i];
                }}
            output_1[i] += actor_encoder_feed_forward_0_bias[i];
            output_1[i] = tanhf(output_1[i]);
        }}
    '''
    for_loops.append(output_for_loop)

    # the last hidden layer which is supposed to have no non-linearity
    n = num_layers - 1
    output_for_loop = f'''
        for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
            output_{str(n)}[i] = 0;
            for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
            }}
            output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
        }}
        '''
    for_loops.append(output_for_loop)

    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for code in for_loops:
        method += code
    # 根据条件决定是否进入当前分支。
    if 'self' in prefix:
        # assign network outputs to control
        assignment = """
        control_n->thrust_0 = output_""" + str(n) + """[0];
        control_n->thrust_1 = output_""" + str(n) + """[1];
        control_n->thrust_2 = output_""" + str(n) + """[2];
        control_n->thrust_3 = output_""" + str(n) + """[3];	
    """
        method += assignment
    # closing bracket
    method += """}\n\n"""
    # 返回当前函数的结果。
    return method


# 定义函数 `neighbor_encoder_c_string`。
def neighbor_encoder_c_string(prefix: str, weight_names: List[str], bias_names: List[str]):
    # 保存或更新 `method` 的值。
    method = """void neighborEmbedder(const float neighbor_inputs[NEIGHBORS * NBR_DIM]) {
    """
    num_layers = len(weight_names)

    # write the for loops for forward-prop
    for_loops = []
    input_for_loop = f'''
            for (int i = 0; i < {prefix}_structure[0][1]; i++) {{
                {prefix}_output_0[i] = 0; 
                for (int j = 0; j < {prefix}_structure[0][0]; j++) {{
                    {prefix}_output_0[i] += neighbor_inputs[j] * actor_encoder_neighbor_embed_layer_0_weight[j][i]; 
                }}
                {prefix}_output_0[i] += actor_encoder_neighbor_embed_layer_0_bias[i];
                {prefix}_output_0[i] = tanhf({prefix}_output_0[i]);
            }}
    '''
    for_loops.append(input_for_loop)

    # hidden layers
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for n in range(1, num_layers - 1):
        for_loop = f'''
                for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
                    {prefix}_output_{str(n)}[i] = 0;
                    for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                        output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
                    }}
                    output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
                    output_{str(n)}[i] = tanhf(output_{str(n)}[i]);
                }}
        '''
        for_loops.append(for_loop)

    # the last hidden layer which is supposed to have no non-linearity
    n = num_layers - 1
    # 根据条件决定是否进入当前分支。
    if n > 0:
        output_for_loop = f'''
                for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
                    output_{str(n)}[i] = 0;
                    for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                        output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
                    }}
                    output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
                    neighbor_embeds[i] += output_{str(n)}[i]; 
                }}
            }}
        '''
        # 调用 `append` 执行当前处理。
        for_loops.append(output_for_loop)

    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for code in for_loops:
        # 保存或更新 `method` 的值。
        method += code
    # method closing bracket
    # 保存或更新 `method` 的值。
    method += """}\n\n"""
    # 返回当前函数的结果。
    return method


# 定义函数 `obstacle_encoder_c_str`。
def obstacle_encoder_c_str(prefix: str, weight_names: List[str], bias_names: List[str]):
    # 保存或更新 `method` 的值。
    method = f"""void obstacleEmbedder(const float obstacle_inputs[OBST_DIM]) {{
        //reset embeddings accumulator to zero
        memset(obstacle_embeds, 0, sizeof(obstacle_embeds));
        
    """
    num_layers = len(weight_names)

    # write the for loops for forward-prop
    for_loops = []
    input_for_loop = f'''
        for (int i = 0; i < {prefix}_structure[0][1]; i++) {{
            {prefix}_output_0[i] = 0;
            for (int j = 0; j < {prefix}_structure[0][0]; j++) {{
                {prefix}_output_0[i] += obstacle_inputs[j] * {weight_names[0].replace('.', '_')}[j][i];
            }}
            {prefix}_output_0[i] += {bias_names[0].replace('.', '_')}[i];
            {prefix}_output_0[i] = tanhf({prefix}_output_0[i]);
        }}
    '''
    for_loops.append(input_for_loop)

    # rest of the hidden layers
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for n in range(1, num_layers - 1):
        for_loop = f'''
            for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
                output_{str(n)}[i] = 0;
                for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                    output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
                }}
                output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
                output_{str(n)}[i] = tanhf(output_{str(n)}[i]);
            }}
        '''
        for_loops.append(for_loop)

    # the last hidden layer which is supposed to have no non-linearity
    n = num_layers - 1
    # 根据条件决定是否进入当前分支。
    if n > 0:
        output_for_loop = f'''
            for (int i = 0; i < {prefix}_structure[{str(n)}][1]; i++) {{
                output_{str(n)}[i] = 0;
                for (int j = 0; j < {prefix}_structure[{str(n)}][0]; j++) {{
                    output_{str(n)}[i] += output_{str(n - 1)}[j] * {weight_names[n].replace('.', '_')}[j][i];
                }}
                output_{str(n)}[i] += {bias_names[n].replace('.', '_')}[i];
                obstacle_embeds[i] += output_{str(n)}[i];
            }}
        '''
        for_loops.append(output_for_loop)

    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for code in for_loops:
        method += code
    # closing bracket
    method += """}\n\n"""
    return method


# 定义函数 `generate_c_model_attention`。
def generate_c_model_attention(model: nn.Module, output_path: str, output_folder: str, testing=False):
    info = generate_c_weights_attention(model, transpose=True)
    model_state_dict = model.state_dict()

    source = ""
    structures = ""
    methods = ""

    # setup all the encoders
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for enc_name, data in info['encoders'].items():
        # data contains [weight_names, bias_names, weights, biases]
        structure = f'static const int {enc_name}_structure [' + str(int(len(data[0]))) + '][2] = {'

        weight_names, bias_names = data[0], data[1]
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for w_name, b_name in zip(weight_names, bias_names):
            w = model_state_dict[w_name].T
            structure += '{' + str(w.shape[0]) + ', ' + str(w.shape[1]) + '},'

        # complete the structure array
        # get rid of the comma after the last curly bracket
        structure = structure[:-1]
        structure += '};\n'
        structures += structure

        method = ""
        # 根据条件决定是否进入当前分支。
        if 'self' in enc_name:
            method = self_encoder_attn_c_str(enc_name, weight_names, bias_names)
        # 当上一分支不满足时，继续判断新的条件。
        elif 'nbr' in enc_name:
            method = neighbor_encoder_c_string(enc_name, weight_names, bias_names)
        # 当上一分支不满足时，继续判断新的条件。
        elif 'obst' in enc_name:
            method = obstacle_encoder_c_str(enc_name, weight_names, bias_names)
        # 当前置条件都不满足时，执行兜底分支。
        else:
            # attention
            method = attention_body

        methods += method

    # headers
    source += headers_network_evaluate if not testing else headers_evaluation
    source += headers_multi_agent_attention

    # helper funcs
    source += linear_activation
    source += sigmoid_activation
    source += relu_activation

    # network eval func
    source += structures
    outputs = info['outputs']
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for output in outputs:
        source += output

    encoders = info['encoders']

    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for key, vals in encoders.items():
        weights, biases = vals[-2], vals[-1]
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for w in weights:
            source += w
        # 遍历当前序列或迭代器，逐项执行下面的逻辑。
        for b in biases:
            source += b

    source += methods

    # 根据条件决定是否进入当前分支。
    if testing:
        source += multi_drone_attn_eval

    # 根据条件决定是否进入当前分支。
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 根据条件决定是否进入当前分支。
    if output_path:
        # 使用上下文管理器包裹后续资源操作。
        with open(output_path, 'w') as f:
            f.write(source)
        f.close()

    return source


# 定义函数 `generate_c_model`。
def generate_c_model(model: nn.Module, output_path: str, output_folder: str, testing=False):
    layer_names, bias_names, weights, biases, outputs = generate_c_weights(model, transpose=True)
    num_layers = len(layer_names)

    structure = 'static const int structure [' + str(int(num_layers)) + '][2] = {'
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for name, param in model.named_parameters():
        param = param.T
        # 根据条件决定是否进入当前分支。
        if 'weight' in name and 'critic' not in name and 'layer_norm' not in name:
            structure += '{' + str(param.shape[0]) + ', ' + str(param.shape[1]) + '},'

    # complete the structure array
    # get rid of the comma after the last curly bracket
    structure = structure[:-1]
    structure += '};\n'

    # write the for loops for forward-prop
    for_loops = []
    input_for_loop = f'''
        for (int i = 0; i < structure[0][1]; i++) {{
            output_0[i] = 0;
            for (int j = 0; j < structure[0][0]; j++) {{
                output_0[i] += state_array[j] * {layer_names[0]}[j][i];
            }}
            output_0[i] += {bias_names[0]}[i];
            output_0[i] = tanhf(output_0[i]);
        }}
    '''
    for_loops.append(input_for_loop)

    # rest of the hidden layers
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for n in range(1, num_layers - 1):
        for_loop = f'''
        for (int i = 0; i < structure[{str(n)}][1]; i++) {{
            output_{str(n)}[i] = 0;
            for (int j = 0; j < structure[{str(n)}][0]; j++) {{
                output_{str(n)}[i] += output_{str(n - 1)}[j] * {layer_names[n]}[j][i];
            }}
            output_{str(n)}[i] += {bias_names[n]}[i];
            output_{str(n)}[i] = tanhf(output_{str(n)}[i]);
        }}
        '''
        for_loops.append(for_loop)

    # the last hidden layer which is supposed to have no non-linearity
    n = num_layers - 1
    output_for_loop = f'''
                for (int i = 0; i < structure[{str(n)}][1]; i++) {{
                    output_{str(n)}[i] = 0;
                    for (int j = 0; j < structure[{str(n)}][0]; j++) {{
                        output_{str(n)}[i] += output_{str(n - 1)}[j] * {layer_names[n]}[j][i];
                    }}
                    output_{str(n)}[i] += {bias_names[n]}[i];
                }}
    '''
    for_loops.append(output_for_loop)

    # assign network outputs to control
    assignment = """
            control_n->thrust_0 = output_""" + str(n) + """[0];
            control_n->thrust_1 = output_""" + str(n) + """[1];
            control_n->thrust_2 = output_""" + str(n) + """[2];
            control_n->thrust_3 = output_""" + str(n) + """[3];	
    """

    # construct the network evaluate function
    controller_eval = """void networkEvaluate(struct control_t_n *control_n, const float *state_array) {"""
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for code in for_loops:
        controller_eval += code
    # assignment to control_n
    controller_eval += assignment

    # closing bracket
    controller_eval += """}"""

    # combine all the codes
    source = ""
    # headers
    source += headers_network_evaluate if not testing else headers_evaluation
    # helper funcs
    source += linear_activation
    source += sigmoid_activation
    source += relu_activation
    # network eval func
    source += structure
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for output in outputs:
        source += output
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for weight in weights:
        source += weight
    # 遍历当前序列或迭代器，逐项执行下面的逻辑。
    for bias in biases:
        source += bias
    source += controller_eval

    # 根据条件决定是否进入当前分支。
    if testing:
        source += single_drone_eval

    # 根据条件决定是否进入当前分支。
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 根据条件决定是否进入当前分支。
    if output_path:
        # 使用上下文管理器包裹后续资源操作。
        with open(output_path, 'w') as f:
            f.write(source)
        f.close()

    return source


# 根据条件决定是否进入当前分支。
if __name__ == '__main__':
    # example use case
    # cfg = AttrDict({
    #     'torch_model_dir': 'swarm_rl/sim2real/torch_models/single',
    #     'output_dir': 'swarm_rl/sim2real/c_models',
    #     'output_model_name': 'model.c',
    #     'testing': True,
    #     'model_type': 'single',
    # })
    # torch_to_c_model(cfg)

    cfg = parse_args()
    torch_to_c_model(args=cfg)
