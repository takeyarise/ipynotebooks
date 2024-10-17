# src: https://github.com/SamsungLabs/ASAM/blob/master/asam.py
import torch
from collections import defaultdict

class ASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        """__init__ _summary_

        Parameters
        ----------
        optimizer : _type_
            _description_
        model : _type_
            _description_
        rho : float, optional
            _description_, by default 0.5
        eta : float, optional
            _description_, by default 0.01

        Example
        -------
        >>> ### in train step ###
        >>> # Ascent Step
        >>> predictions = model(inputs)
        >>> batch_loss = criterion(predictions, targets)
        >>> batch_loss.mean().backward()
        >>> minimizer.ascent_step()
        >>> # Descent Step
        >>> criterion(model(inputs), targets).mean().backward()
        >>> minimizer.descent_step()
        """
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()


class SAM(ASAM):
    def __init__(self, optimizer, model, rho=0.05):
        """__init__ _summary_

        Parameters
        ----------
        optimizer : _type_
            _description_
        model : _type_
            _description_
        rho : float, optional
            _description_, by default 0.5

        Example
        -------
        >>> ### in train step ###
        >>> # Ascent Step
        >>> predictions = model(inputs)
        >>> batch_loss = criterion(predictions, targets)
        >>> batch_loss.mean().backward()
        >>> minimizer.ascent_step()
        >>> # Descent Step
        >>> criterion(model(inputs), targets).mean().backward()
        >>> minimizer.descent_step()
        """
        super().__init__(optimizer, model, rho, 0)

    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()
