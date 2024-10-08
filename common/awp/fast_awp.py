import torch


class AWP:
    def __init__(self, model, optimizer, *, adv_param='weight',
                 adv_lr=0.001, adv_eps=0.001):
        """AWP

        src: https://www.kaggle.com/code/junkoda/fast-awp

        Parameters
        ----------
        model : _type_
            _description_
        optimizer : _type_
            _description_
        adv_param : str, optional
            _description_, by default 'weight'
        adv_lr : float, optional
            _description_, by default 0.001
        adv_eps : float, optional
            _description_, by default 0.001

        Example
        -------
        >>> awp = AWP(model, optimizer, adv_lr=0.001, adv_eps=0.001)
        >>> awp_start = 1.0
        >>> for epoch in range(epochs):
        >>>     for x, y in enumerate(loader_train):
        >>>         if epoch >= awp_start:
        >>>             awp.perturb(input_ids, attention_mask, y, criterion)
        >>>         y_pred = model(x)
        >>>         loss = criterion(y_pred, y)
        >>>         loss.backward()
        >>>         awp.restore()
        >>>         optimizer.zero_grad()
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def perturb(self, input_ids, attention_mask, y, criterion):
        """
        Perturb model parameters for AWP gradient
        Call before loss and loss.backward()
        """
        self._save()  # save model parameters
        self._attack_step()  # perturb weights

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                grad = self.optimizer.state[param]['exp_avg']
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.detach())

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Set lower and upper limit in change
                    limit_eps = self.adv_eps * param.detach().abs()
                    param_min = param.data - limit_eps
                    param_max = param.data + limit_eps

                    # Perturb along gradient
                    # w += (adv_lr * |w| / |grad|) * grad
                    param.data.add_(grad, alpha=(self.adv_lr * (norm_data + e) / (norm_grad + e)))

                    # Apply the limit to the change
                    param.data.clamp_(param_min, param_max)

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self):
        """
        Restore model parameter to correct position; AWP do not perturbe weights, it perturb gradients
        Call after loss.backward(), before optimizer.step()
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
