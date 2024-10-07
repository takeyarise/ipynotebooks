import copy
import dataclasses
from functools import partial
from typing import Optional, Union, Callable, Self, Tuple, Dict, Iterable
import numpy as np
from . import (
    jitter, scaling, rotation, permutation, magnitude_warping, time_warping,
    dropout, flipping, cutout, shift, window_warping, window_slicing,
)


__all__ = [
    'RandAugment',
    'UCR_POLICY',
]


@dataclasses.dataclass(frozen=True)
class Range:
    name: str
    min: float
    max: float
    bins: Optional[int] = dataclasses.field(default=None)
    range: Optional[np.ndarray] = dataclasses.field(default=None)

    def get_range(self, bins: int) -> np.ndarray:
        if self.range is None:
            return np.linspace(self.min, self.max, bins)
        else:
            return self.range


@dataclasses.dataclass(frozen=True)
class Select:
    name: str
    values: tuple
    bins: Optional[int] = dataclasses.field(default=None)
    range: Optional[np.ndarray] = dataclasses.field(default=None)

    def get_range(self, bins: int) -> np.ndarray:
        if self.range is None:
            values = np.array(self.values)
            return values[np.linspace(0, len(values), bins + 1, dtype=int)[:-1]]
        else:
            return self.range


@dataclasses.dataclass(frozen=True)
class Policy:
    name: str
    args: Optional[Tuple[Union[Select, Range], ...]]
    function_map: Dict[str, Callable] = dataclasses.field(default_factory=lambda: {
        'Identity': lambda x: x,
        'Jitter': jitter,
        'Scaling': scaling,
        'MagnitudeWarp': magnitude_warping,
        'TimeWarp': time_warping,
        'Permutation': permutation,
        'Rotation': rotation,
        'Dropout': dropout,
        'Flipping': flipping,
        'Cutout': cutout,
        'Shift': shift,
        'WindowWarp': window_warping,
        'WindowSlice': window_slicing,
    }, init=False, repr=False)

    def build_range(self, bins: int) -> Self:
        if self.args is None:
            return self
        new_args = tuple(
            dataclasses.replace(a, bins=bins, range=a.get_range(bins))
            for a in self.args
        )
        return dataclasses.replace(self, args=new_args)

    def get_transform(self, bins: int, magnitude: Union[int,list]) -> Callable:
        assert self.name in self.function_map.keys(), f'"{self.name}" is not implemented.'
        if self.args is None:
            return self.function_map[self.name]
        else:
            if isinstance(magnitude, int):
                magnitude = [magnitude] * len(self.args)
            args = {
                v.name: v.get_range(bins)[m]
                for v, m in zip(self.args, magnitude)
            }
            return partial(self.function_map[self.name], **args)


UCR_POLICY = (  # paper: https://arxiv.org/abs/2102.08310
    Policy('Identity', None),
    Policy('Jitter', (Range('sigma', 0.01, 0.5),)),
    Policy('TimeWarp', (Select('knot', (3, 4, 5)), Range('sigma', 0.01, 0.5))),
    Policy('WindowSlice', (Range('ratio', 0.95, 0.6),)),
    Policy('WindowWarp', (Select('ratio', (0.1,)), Range('scale', 0.1, 2.))),
    Policy('Scaling', (Range('sigma', 0.1, 2.0),)),
    Policy('MagnitudeWarp', (Select('knot', (3, 4, 5)), Range('sigma', 0.1, 2.0))),
    Policy('Permutation', (Select('max_segments', (3, 4, 5, 6)),)),
    Policy('Dropout', (Range('p', 0.05, 0.5),)),
)


RANDECG_POLICY = (
    # paper: https://www.jstage.jst.go.jp/article/pjsai/JSAI2021/0/JSAI2021_2N1IS2a02/_pdf/-char/ja
    Policy('Identity', None),
    Policy('Scaling', (Range('sigma', 0.25, 4),)),
    Policy('Flipping', (Range('p', 0.0, 1.0),)),
    Policy('Dropout', (Range('p', 0.0, 0.4),)),
    Policy('Shift', (Range('ratio', 0.0, 1.0),)),
    Policy('Cutout', (Range('ratio', 0.0, 0.4),)),
    Policy('Sine', (Range('amp', 0.0, 1.0), Range('freq', 0.001, 0.02))),  # FIXME
    Policy('SquarePulse', (Range('amp', 0.0, 0.02), Range('freq', 0.001, 0.1))),  # FIXME
    Policy('Jitter', (Range('sigma', 0.0, 0.02),)),
    Policy('SinePartial', (Range('length', 0.0, 0.1), Range('freq', 0.1, 1.0))),  # FIXME
    Policy('SquarePulsePartial', (Range('length', 0.0, 1.0), Range('freq', 0.02, 1.0))),  # FIXME
    Policy('GaussianNoisePartial', (Range('length', 0, 0.1), Range('sigma', 0.0, 0.02))),  # FIXME
    Policy('FIRLowpass', (Range('freq', 0.1, 0.4999),)),  # FIXME
)


class RandAugment:
    def __init__(self, magnitude: Union[int,str], bins: int=20, num_ops: int=2, policy: Union[tuple,list]=UCR_POLICY):
        """RandAugment

        paper: https://arxiv.org/abs/1909.13719

        Parameters
        ----------
        magnitude : Union[int, str]
            if magnitude is int, 1 <= magnitude <= bins
        bins : int, optional
            bins, by default 20
        num_ops: int
            by default 2
        policy : Iterable, optional
            policy, by default UCR_POLICY
        """
        if isinstance(magnitude, int):
            assert 1 <= magnitude <= bins, f'1 <= magnitude <= {bins}, {magnitude=}'
            magnitude = magnitude - 1
        elif isinstance(magnitude, str):
            assert magnitude in ('random')
        self.bins = 20
        self.magnitude = magnitude
        self.policy = tuple(
            p.build_range(bins)
            for p in policy
        )
        self.num_transforms = len(self.policy)
        self.num_ops = num_ops

    def __call__(self, x):
        ids = np.random.choice(np.arange(self.num_transforms), self.num_ops, replace=False)
        if isinstance(self.magnitude, int):
            for i in ids:
                x = self.policy[i].get_transform(self.bins, self.magnitude)(x)
            return x
        else:
            for i in ids:
                x = self.policy[i].get_transform(self.bins, np.random.choice(self.bins))(x)
            return x

    def __repr__(self) -> str:
        s = (
            f'{self.__class__.__name__}('
            f'num_ops={self.num_ops}'
            f', magnitude={self.magnitude}'
            f', bins={self.bins}'
            f', policy={self.policy}'
            f')'
        )
        return s


class WAugment:
    # TODO: loader 後の shape が (-1, N, 3, W) である可能性があるため確認すること
    def __init__(self, magnitude: int, bins=20, policy: Union[tuple,list]=UCR_POLICY):
        """W- or alpha-augment

        paper: https://arxiv.org/abs/2102.08310

        Parameters
        ----------
        magnitude : int
            1 <= magnitude <= bins
        bins : int, optional
            bins, by default 20
        policy : Iterable, optional
            policy, by default UCR_POLICY
        """
        assert 1 <= magnitude <= bins, f'1 <= magnitude <= {bins}, {magnitude=}'
        self.bins = 20
        self.magnitude = magnitude - 1
        self.policy = tuple(
            p.build_range(bins)
            for p in policy
        )
        self.transforms = tuple(
            p(bins, magnitude)
            for p in self.policy
        )

    def __call__(self, x):
        return np.array([func(x) for func in self.transforms])

    def __repr__(self) -> str:
        s = (
            f'{self.__class__.__name__}('
            f', magnitude={self.magnitude}'
            f', bins={self.bins}'
            f', policy={self.policy}'
            f')'
        )
        return s


class CTAugment:
    def __init__(self, bins=20, num_ops=2, threthold=0.8, decay=0.99, policy: Union[tuple,list]=UCR_POLICY):
        """
        paper: https://arxiv.org/abs/1911.09785
        """
        self.bins = bins
        self.num_ops = num_ops
        self.threthold = threthold
        self.decay = decay
        self.policy = tuple(
            p.build_range(bins)
            for p in policy
        )
        self.num_policies = len(policy)
        self.rate = [
            [np.ones(bins)] * len(p.args)
            for p in self.policy
        ]
        self.reset_selected()

    def reset_selected(self):
        self.selected_transforms = list()
        self.selected_magnitudes = list()

    def __call__(self, x):
        selected = np.random.choice(np.arange(len(self.policy)), self.num_ops, replace=False)
        self.selected_transforms.extend(selected.tolist())
        mgs = list()
        for i in selected:
            rates = copy.deepcopy(self.rate[i])
            for i in range(rates):
                rates[i][rates[i] < self.threthold] = 0
                rates[i] = rates[i] / rates[i].sum()
            magnitudes = [np.random.choice(len(r), p=r) for r in rates]
            transrate = self.policy[i].get_transform(self.bins, magnitudes)
            x = transrate(x)
            mgs.append(magnitudes)
        self.selected_magnitudes.extend(mgs)
        return x

    def update_rate(self, mean_pred_errors):
        """ update rate

        Parameters
        ----------
        mean_pred_errors : _type_
            $1-\frac{1}{2L}\sum |\mathrm{pred}-\mathrm{label}|$, `1 - (pred - true).abs().mean(dim=-1).mean(dim=-1)` かも
        """
        assert (self.num_ops * mean_pred_errors.size(0) == len(self.selected_transforms))

        for ii, idx in enumerate(self.selected_transforms):
            for i, m in enumerate(self.selected_magnitudes):
                current = self.rate[idx][i][m]
                self.rate[idx][i][m] = current * self.decay + mean_pred_errors[int(ii//self.num_ops)] * (1 - self.decay)
        self.reset_selected()
