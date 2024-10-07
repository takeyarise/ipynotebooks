"""
頑張って下を変更したけどあってたかも
* https://github.com/takeyarise/ipynbs/commit/eff3badca0ea4e8a564422d0096497b154ea6fc6

参考:
* https://github.com/logancyang/loss-landscape-anim/tree/master
* https://github.com/marcellodebernardi/loss-landscapes/tree/master
* https://github.com/tomgoldstein/loss-landscape/tree/master
"""
import copy
import torch
import torch.nn as nn
from .params import (
    get_random_state,
    get_diff_state,
    create_random_direction,
    orthogonal_state,
	normalize_direction,
	create_pca_direction,
	trajectory_points,
)


@torch.no_grad()
def calc_linear_interpolation(src_net, dst_net, loss_fn, start=0., end=1., num_steps=100):
	""" Calculate linear interpolation.

	Parameters
	----------
	src_net : nn.Module
		source network
	dst_net : nn.Module
		destination network
	loss_fn : callable[nn.Module -> float]
		the function to get loss from network
	start : int, optional
		start point, by default 0
	end : int, optional
		end point, by default 1
	num_steps : int, optional
		number of steps, by default 100
	ignore_bias_and_bn : bool, optional
		ignore bias and batch normalization, by default True

	Returns
	-------
	(list, torch.Tensor)
		losses and x coordinates

	Examples
	--------
	>>> def loss_fn(net):
	...     net.to('cuda').train()
	...     return net(torch.randn(10)).sum().item()
	>>>
	>>> y, x = calc_linear_interpolation(src_net, dst_net, loss_fn, start=0, end=1, num_steps=100)
	"""
	assert start <= end
	assert num_steps >= 1
	assert isinstance(src_net, nn.Module)
	assert isinstance(dst_net, nn.Module)
	assert callable(loss_fn)

	src_net.to('cpu')
	dst_net.to('cpu')
	src_net.eval()
	dst_net.eval()

	state = src_net.state_dict()
	direction = get_diff_state(state, dst_net.state_dict())

	net = copy.deepcopy(src_net)
	x_coords = torch.linspace(start, end, num_steps)
	ret = list()
	for alpha in x_coords:
		diff = [v * alpha for v in direction]

		new_states = copy.deepcopy(state)
		for (k, v), d in zip(new_states.items(), diff):
			v.add_(d.type(v.type()))
		net.load_state_dict(new_states)
		loss = loss_fn(net)
		ret.append(loss)

	return ret, x_coords.to('cpu')


@torch.no_grad()
def calc_landscape_1d(net, loss_fn, start=-1., end=1., num_steps=100, direction=None):
	""" calculate 1d landscape.

	Parameters
	----------
	net : nn.Module
		network
	loss_fn : callable[nn.Module -> float]
		the function to get loss from network
	start : int, optional
		start point, by default -1
	end : int, optional
		end point, by default 1
	num_steps : int, optional
		number of steps, by default 100
	direction : OrderedDict, optional
		direction, by default None

	Returns
	-------
	(list, torch.Tensor)
		losses and x coordinates

	Examples
	--------
	>>> def loss_fn(net):
	...     net.to('cuda').train()
	...     return net(torch.randn(10)).sum().item()
	>>>
	>>> y, x = calc_landscape_1d(net, loss_fn, start=-1, end=1, num_steps=100)
	"""
	assert start <= end
	assert num_steps >= 1
	assert isinstance(net, nn.Module)
	assert callable(loss_fn)
	net.eval()
	net.to('cpu')

	if direction is None:
		direction = create_random_direction(net, mode='filter', ignore_bias_and_bn=True)

	state = net.state_dict()
	src_net = copy.deepcopy(net)
	x_coords = torch.linspace(start, end, num_steps)
	losses = torch.empty(len(x_coords))
	for i, alpha in enumerate(x_coords):
		diff = [v * alpha for v in direction]

		new_state = copy.deepcopy(state)
		for (k, v), d in zip(new_state.items(), diff):
			v.add_(d.type(v.type()))
		src_net.load_state_dict(new_state)
		loss = loss_fn(src_net)
		losses[i] = loss

	return losses.to('cpu'), x_coords.to('cpu')


@torch.no_grad()
def calc_landscape_2d(net, loss_fn, x_range, y_range, x_direction=None, y_direction=None, is_y_random=True):
	""" calculate 2d landscape.

	Parameters
	----------
	net : nn.Module
		network
	loss_fn : callable[nn.Module -> float]
		the function to get loss from network.
	x_range : tuple
		x range, (start, end [,num_steps])
	y_range : tuple
		y range, (start, end [,num_steps])
	x_direction : OrderedDict, optional
		x direction, by default None
	y_direction : OrderedDict, optional
		y direction, by default None
	is_y_random : bool, optional
		is y random, by default True

	Returns
	-------
	(torch.Tensor, torch.Tensor, torch.Tensor)
		losses, x coordinates, y coordinates

	Examples
	--------
	>>> def loss_fn(net):
	...     net.to('cuda').train()
	...     return net(torch.randn(10)).sum().item()
	>>>
	>>> y, x, z = calc_landscape_2d(net, loss_fn, x_range=(-1, 1, 5), y_range=(-1, 1, 5))
	"""
	assert isinstance(net, nn.Module)
	assert callable(loss_fn)
	assert len(x_range) == 3 and len(y_range) == 3
	net.eval()
	net.to('cpu')

	x_coords = torch.linspace(*x_range)
	y_coords = torch.linspace(*y_range)

	if x_direction is None:
		x_direction = create_random_direction(net, mode='filter', ignore_bias_and_bn=True)
	if y_direction is None:
		if is_y_random:
			y_direction = create_random_direction(net, mode='filter', ignore_bias_and_bn=True)
		else:
			y_direction = get_random_state(net.state_dict())
			orthogonal_state(y_direction, x_direction)
			normalize_direction(y_direction, net, mode='filter', ignore_bias_and_bn=True)

	src_net = copy.deepcopy(net)
	state = net.state_dict()
	losses = torch.empty(len(x_coords), len(y_coords))
	for i, alpha in enumerate(x_coords):
		x_diff = [v * alpha for v in x_direction]
		for j, beta in enumerate(y_coords):
			y_diff = [v * beta for v in y_direction]

			new_state = copy.deepcopy(state)
			for (k, v), dx, dy in zip(new_state.items(), x_diff, y_diff):
				v.add_(dx.type(v.type()))
				v.add_(dy.type(v.type()))
			src_net.load_state_dict(new_state)
			loss = loss_fn(src_net)
			losses[i, j] = loss

	return losses.to('cpu'), x_coords.to('cpu'), y_coords.to('cpu')


@torch.no_grad()
def calc_trajectory(net, trajectory_states):
	""" calculate trajectory.

	Parameters
	----------
	net : nn.Module
		network
	trajectory_states : list[OrderedDict]
		trajectory states, [init_net.state_dict(), ..., final_net.state_dict()]

	Returns
	-------
	(torch.Tensor, torch.Tensor, dict)
		x coords, y coords, pca result

	Examples
	--------
	>>> x, y, pca_res = calc_trajectory(net, trajectory_states)
	"""
	net.to('cpu')
	x_direction, y_direction, pca_res = create_pca_direction(net, trajectory_states)
	x, y = trajectory_points(x_direction, y_direction, net, trajectory_states)
	return x, y, pca_res
