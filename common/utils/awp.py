"""
paper: https://arxiv.org/abs/2004.05884
"""
import torch


class AWP:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        adv_param="weight",
        adv_lr=1.0,
        adv_eps=0.01,
    ):
        """Adversarial Weight Perturbation (AWP)

        src: https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332492#1828747

        Parameters
        ----------
        model : torch.nn.Module
            model
        criterion : torch.nn.Module
            loss function
        optimizer : torch.optim.Optimizer
            optimizer
        adv_param : str, optional
            parameter name to perturb, by default "weight"
        adv_lr : float, optional
            learning rate of perturbation, by default 1.0
        adv_eps : float, optional
            perturbation size, by default 0.01

        Examples
        --------
        >>> awp = AWP(model, criterion, optimizer)
        >>> # train loop
        >>> for inputs, label in loader:
        >>>     loss = criterion(model(inputs), label)
        >>>     optimizer.zero_grad()
        >>>     loss.backward()
        >>>     # ---
        >>>     loss = awp.attack_backward(inputs, label)
        >>>     loss.backward()
        >>>     awp.restore()
        >>>     # ---
        >>>     optimizer.step()
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs, label):
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            self._save()
            self._attack_step()  # モデルを近傍の悪い方へ改変
            y_preds = self.model(inputs)
            adv_loss = self.criterion(y_preds.view(-1, 1), label.view(-1, 1))
            mask = label.view(-1, 1) != -1
            adv_loss = torch.masked_select(adv_loss, mask).mean()
            self.optimizer.zero_grad()
        return adv_loss

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
