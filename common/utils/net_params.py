from pathlib import Path
import torch
import torch.nn as nn


def save_net(net: nn.Module, path: Path):
    """ save net parameters

    ```python
    torch.save(net.to('cpu').state_dict(), path)
    ```

    Parameters
    ----------
    net : torch.nn.Module
        net
    path : str
        path

    Examples
    --------
    >>> save_net_params(net, 'net.pth')
    """
    torch.save(net.to('cpu').state_dict(), str(path))


def load_net(net: nn.Module, path: Path):
    """ load net parameters

    ```python
    net.to('cpu')
    net.load_state_dict(torch.load(path))
    ```

    Parameters
    ----------
    net : torch.nn.Module
        net
    path : str
        path

    Examples
    --------
    >>> load_net_params(net, 'net.pth')
    """
    net.to('cpu')
    net.load_state_dict(torch.load(str(path)))


def setup_init_net(net: nn.Module, path: Path):
    """ setup init net

    ```python
    if path.exists(): load_net(net, path)
    else: save_net(net, path)
    ```

    Parameters
    ----------
    net : torch.nn.Module
        net
    path : str
        path

    Examples
    --------
    >>> setup_init_net(net, 'net.pth')
    """
    if path.exists():
        load_net(net, path)
    else:
        save_net(net, path)
