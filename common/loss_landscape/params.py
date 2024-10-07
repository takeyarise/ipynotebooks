import copy
from collections import OrderedDict
from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters


def get_random_state(state: OrderedDict):
    return [torch.randn(v.size()) for v in state.values()]


def get_diff_state(src_state: OrderedDict, dst_state: OrderedDict):
    return [v2 - v1 for v1, v2 in zip(src_state.values(), dst_state.values())]


def normalize_direction(direction, net: nn.Module, mode: str='filter', ignore_bias_and_bn: bool=True):
    assert mode in ['filter', 'layer', 'dir_filter', 'dir_layer'], 'mode must be one of [filter, layer, dir_filter, dir_layer]'
    for d, (k, w) in zip(direction, net.state_dict().items()):
        if d.dim() <= 1:
            if ignore_bias_and_bn:
                d.fill_(0)
            else:
                d.copy_(w)
            continue

        if mode == 'filter':
            for dd, ww in zip(d, w):
                dd.mul_(ww.norm() / dd.norm() + 1e-10)
        elif mode == 'layer':
            d.mul_(w.norm() / d.norm())
        elif mode == 'dir_filter':
            for dd in d:
                dd.div_(dd.norm() + 1e-10)
        elif mode == 'dir_layer':
            d.div_(d.norm())


def orthogonal_state(direction, base):
    for d, b in zip(direction, base):
        if d.dim() <= 1:
            continue
        d.sub_(d.dot(b) * b)
        d.div_(d.norm() + 1e-10)


def set_bias_and_bn_to_zero(params):
    for d in params:
        if d.dim() <= 1:
            d.fill_(0)


def create_random_direction(net: nn.Module, mode: str='filter', ignore_bias_and_bn: bool=True):
    assert mode in ['filter', 'layer', 'dir_filter', 'dir_layer'], 'norm must be one of [filter, layer, dir_filter, dir_layer]'
    direction = get_random_state(net.state_dict())
    normalize_direction(direction, net, mode=mode, ignore_bias_and_bn=ignore_bias_and_bn)
    return direction


def create_target_direction(src: nn.Module, dst: nn.Module, mode: str='filter'):
    assert mode in ['filter', 'layer', 'dir_filter', 'dir_layer'], 'norm must be one of [filter, layer, dir_filter, dir_layer]'
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    diff_state = get_diff_state(src_state, dst_state)
    return diff_state


def projection(weight_vector, x_vector, y_vector):
    def projection_1d(weight_vector, vector):
        scale = torch.dot(weight_vector, vector) / vector.norm()
        return scale.item()

    x = projection_1d(weight_vector, x_vector)
    y = projection_1d(weight_vector, y_vector)
    return x, y


def trajectory_point(x_direction, y_direction, src_net, dst_net):
    dx = parameters_to_vector(x_direction)
    dy = parameters_to_vector(y_direction)
    diff = get_diff_state(src_net.state_dict(), dst_net.state_dict())
    diff = parameters_to_vector(diff)
    x, y = projection(diff, dx, dy)
    return x, y


def trajectory_points(x_direction, y_direction, net, trajectory_states):
    net.to('cpu')
    src_net = copy.deepcopy(net)
    dst_net = copy.deepcopy(net)
    x_coords = list()
    y_coords = list()
    for i in range(1, len(trajectory_states)):
        src_net.load_state_dict(trajectory_states[i-1])
        dst_net.load_state_dict(trajectory_states[i])
        x, y = trajectory_point(x_direction, y_direction, src_net, dst_net)
        x_coords.append(x)
        y_coords.append(y)
    return np.array(x_coords), np.array(y_coords)


def create_pca_direction(net, trajectory_states):
    net.to('cpu')
    matrix = list()
    for i in range(1, len(trajectory_states)):
        diff = get_diff_state(trajectory_states[i-1], trajectory_states[i])
        set_bias_and_bn_to_zero(diff)
        diff = parameters_to_vector(diff)
        matrix.append(diff.numpy().copy())

    pca = PCA(n_components=2)
    pca.fit(np.array(matrix))
    pc1 = torch.from_numpy(np.array(pca.components_[0]))
    pc2 = torch.from_numpy(np.array(pca.components_[1]))

    ret = dict(
        explained_variance_ratio=pca.explained_variance_ratio_,
        singular_values=pca.singular_values_,
        explained_variance=pca.explained_variance_,
    )

    vector_to_parameters(pc1, net.parameters())
    x_direction = copy.deepcopy(list(net.parameters()))
    vector_to_parameters(pc2, net.parameters())
    y_direction = copy.deepcopy(list(net.parameters()))
    return x_direction, y_direction, ret
