import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


def train(train_step_fn, train_loader, epochs, scheduler=None, is_tqdm=True, show_interval=None):
    """ training loop

    Parameters
    ----------
    train_step_fn : function
        function to train
    train_loader : torch.utils.data.DataLoader
        train data loader
    device : torch.device
        device
    epochs : int
        epochs
    scheduler : torch.optim.lr_scheduler
        scheduler
    is_tqdm : bool
        use tqdm
    show_interval : int
        show interval

    Returns
    -------
    history : list
        history of loss

    Examples
    --------
    >>> def train_step_fn(batch):
    >>>     x, y = batch
    >>>     x, y = x.to(device), y.to(device)
    >>>     out = net(x)
    >>>     loss = criterion(out, y)
    >>>     optimizer.zero_grad()
    >>>     loss.backward()
    >>>     optimizer.step()
    >>>     return loss.item()
    >>>
    >>> history = train(train_step_fn, train_loader, device, epochs, scheduler)
    """
    history = list()
    iterator = range(epochs)
    if is_tqdm:
        iterator = tqdm(iterator)
    for epoch in iterator:
        total = 0
        total_loss = 0.
        for batch in train_loader:
            loss = train_step_fn(batch)
            total += batch[0].size(0)
            total_loss += loss * batch[0].size(0)

        history.append(total_loss / total)
        if scheduler is not None:
            scheduler.step()

        if show_interval is not None and (epoch == 0 or (epoch + 1) == epochs or (epoch + 1) % show_interval == 0):
            print(f'epoch: {epoch}, loss: {history[-1]}')
    return history


def general_training(train_step_fn, loader, epochs, scheduler=None):
    """ training loop.

    Parameters
    ----------
    train_step_fn : function
        function to train, [batch] -> float, list or dict
    train_loader : torch.utils.data.DataLoader
        train data loader
    epochs : int
        epochs
    scheduler : Option[torch.optim.lr_scheduler]
        scheduler, by default None
    device : Option[torch.device]
        device, by default None

    Returns
    -------
    history : list or dict
        history of return value of train_step_fn

    Examples
    --------
    >>> def train_step_fn(batch):
    >>>     x, y = batch
    >>>     out = net(x)
    >>>     loss = criterion(out, y)
    >>>     optimizer.zero_grad()
    >>>     loss.backward()
    >>>     optimizer.step()
    >>>     return loss.item()
    >>>
    >>> history = general_training(train_step_fn, train_loader, epochs, scheduler)
    """
    assert callable(train_step_fn)

    num_batches = len(loader)
    history = list()
    for _ in tqdm(range(epochs)):
        batch_nums = [0] * num_batches
        rets = [None] * num_batches
        for i, batch in enumerate(loader):
            ret = train_step_fn(batch)
            batch_nums[i] = batch[0].size(0)
            rets[i] = ret
        batch_nums = np.array(batch_nums)
        if isinstance(rets[0], float):
            table = np.sum(np.array(rets) * batch_nums) / np.sum(batch_nums)
        elif isinstance(rets[0], (tuple, list, dict)):
            table = pd.DataFrame(rets).to_dict(orient='list')
            for k, v in table.items():
                table[k] = np.sum(v.values * batch_nums) / np.sum(batch_nums)
        else:
            raise NotImplementedError

        if scheduler is not None:
            scheduler.step()

        history.append(table)
    if isinstance(history[0], float):
        return history
    else:
        return pd.DataFrame(history).to_dict(orient='list')


@torch.no_grad()
def evaluate(net, test_loader, device, evaluate_step_fn=None):
    """ evaluate

    Parameters
    ----------
    net : torch.nn.Module
        network
    test_loader : torch.utils.data.DataLoader
        test data loader
    device : torch.device
        device
    predict_fn : function
        predict function, if None, use argmax

    Returns
    -------
    accuracy : float
        accuracy

    Examples
    --------
    >>> def evaluate_step_fn(batch):
    >>>     x, y = batch
    >>>     x, y = x.to(device), y.to(device)
    >>>     pred = torch.argmax(net(x).data, dim=1)
    >>>     return (pred == y).sum().item()
    >>>
    >>> evaluate(net, test_loader, device, evaluate_step_fn)
    """
    if evaluate_step_fn is None:
        def evaluate_step_fn(batch):
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = torch.argmax(net(x).data, dim=1)
            return (pred == y).sum().item()

    correct = 0
    total = 0

    net.eval()
    for data in test_loader:
        correct += evaluate_step_fn(data)
        total += data[0].size(0)

    return correct / total
