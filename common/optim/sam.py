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
        if isinstance(module, _BatchNorm):
            module.train()
    model.apply(_disable)
    try:
        yield
    finally:
        model.apply(_enable)


class SAM(torch.optim.Optimizer):
    # orig src: https://github.com/davda54/sam
    # - modified to support torch.compile
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
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
                if p.grad is None:
                    continue
                if "old_p" not in self.state[p]:
                    self.state[p]["old_p"] = torch.empty_like(p)
                self.state[p]["old_p"].copy_(p)
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.copy_(self.state[p]["old_p"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm_squares = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grad = (torch.abs(p) if group["adaptive"] else 1.0) * p.grad
                    norm_squares.append(torch.sum(grad ** 2).to(shared_device))
        norm = torch.sqrt(torch.sum(torch.stack(norm_squares)))
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
