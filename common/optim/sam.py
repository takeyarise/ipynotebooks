from contextlib import contextmanager
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model):
    # src: https://github.com/davda54/sam/blob/gsam/example/utility/bypass_bn.py
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    # src: https://github.com/davda54/sam/blob/gsam/example/utility/bypass_bn.py
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


@contextmanager
def bypass_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.eval()
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_training"):
            module.train()
    model.apply(_disable)
    try:
        yield
    finally:
        model.apply(_enable)


class SAM(torch.optim.Optimizer):
    # src: https://github.com/davda54/sam
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """__init__ _summary_

        Parameters
        ----------
        params : _type_
            _description_
        base_optimizer : _type_
            _description_
        rho : float, optional
            _description_, by default 0.05
        adaptive : bool, optional
            _description_, by default False

        Example
        -------
        >>> for batch in dataset.train:
        >>>   inputs, targets = (b.to(device) for b in batch)
        >>>   # first forward-backward step
        >>>   enable_running_stats(model)  # <- this is the important line if you use batch normalization
        >>>   predictions = model(inputs)
        >>>   loss = smooth_crossentropy(predictions, targets)
        >>>   loss.mean().backward()
        >>>   optimizer.first_step(zero_grad=True)
        >>>   # second forward-backward step
        >>>   disable_running_stats(model)  # <- this is the important line if you use batch normalization
        >>>   smooth_crossentropy(model(inputs), targets).mean().backward()
        >>>   optimizer.second_step(zero_grad=True)
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
