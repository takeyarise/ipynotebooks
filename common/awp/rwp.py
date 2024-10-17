# from: https://openreview.net/pdf?id=VcuScWOJfl
import numpy as np
import torch


class RWP:
    """
    Example of RWP usage:
    rwp = RWP()
    x1, x2, y1, y2 = ...  # prepare data

    l1 = criterion(model(x1), y1) * (1 - rwp.alpha) * 2
    l1.backward()

    rwp.perturb(model)
    l2 = criterion(model(x2), y2) * rwp.alpha * 2
    l2.backward()
    rwp.restore(model)

    optimizer.step()
    """
    def __init__(self, *, gamma=0.01, alpha=0.5):
        """

        Parameters
        ----------
        gamma : float, optional
            _description_, by default 0.01
        alpha : float, optional
            _description_, by default 0.5
        """
        self.gamma = gamma
        self.alpha = alpha
        self.noise = None

    def perturb(self, model):
        """
        Perturb model parameters for RWP gradient
        Call before loss and loss.backward()

        Returns:
            weight (float): weight for loss
        """
        ##################### grw #############################
        noise = []
        for mp in model.parameters():
            if len(mp.shape) > 1:
                sh = mp.shape
                sh_mul = np.prod(sh[1:])
                temp = mp.view(sh[0], -1).norm(dim=1, keepdim=True).repeat(1, sh_mul).view(mp.shape)
                temp = torch.normal(0, self.gamma*temp).to(mp.data.device)
            else:
                temp = torch.empty_like(mp, device=mp.data.device)
                temp.normal_(0, self.gamma*(mp.view(-1).norm().item() + 1e-16))
            noise.append(temp)
            mp.data.add_(noise[-1])
        self.noise = noise

    def restore(self, model):
        """
        Restore model parameter to correct position
        Call after loss.backward(), before optimizer.step()
        """
        if self.noise is None:
            return

        with torch.no_grad():
            for mp, noise in zip(model.parameters(), self.noise):
                mp.data.sub_(noise)
            self.noise = None
