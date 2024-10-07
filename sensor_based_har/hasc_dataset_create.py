from typing import List, Tuple
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd


def load_meta(path: Path) -> pd.DataFrame:
    """Function for loading meta data of HASC dataset

    Parameters
    ----------
    path: Path
        Directory path of HASC dataset, which is parent directory of "BasicActivity" directory.

    Returns
    -------
    metas: pd.DataFrame:
        meta data of HASC dataset.
    """

    def replace_meta_str(s: str) -> str:
        s = s.replace('：', ':')
        s = s.replace('TerminalPosition:TerminalPosition:',
                            'TerminalPosition:')
        s = s.replace('sneakers:leathershoes', 'sneakers;leathershoes')
        s = s.replace('Frequency(Hz)', 'Frequency')
        return s

    def read_meta(file: Path) -> dict:
        with file.open(mode='r', encoding='utf-8') as f:
            ret = [s.strip() for s in f.readlines()]
            ret = filter(bool, ret)
            ret = [replace_meta_str(s) for s in ret]
            ret = [e.partition(':')[0::2] for e in ret]
            ret = {key.strip(): val.strip() for key, val in ret}
        act, person, file_name = file.parts[-3:]
        ret['act'] = act
        ret['person'] = person
        ret['file'] = file_name.split('.')[0]
        return ret

    path = path / 'BasicActivity'
    assert path.exists(), '{} is not exists.'.format(str(path))
    files = path.glob('**/**/*.meta')
    metas = list(map(read_meta, files))
    metas = pd.DataFrame(metas)

    acts = enumerate(metas['act'].unique())
    metas['act_id'] = metas['act'].map(dict(acts))
    return metas


def mask_meta(metas: pd.DataFrame) -> pd.DataFrame:
    """Function for making meta mask of HASC dataset

    Parameters
    ----------
    metas: pd.DataFrame
        meta data of HASC dataset.

    Returns
    -------
    masked_metas: list[bool]:
        mask of meta data of HASC dataset.
    """
    mask = [
        metas['Frequency'] == '100',
        metas['TerminalType'].str.contains('iP'),
        metas['act'] != '0_sequence',
    ]
    return np.logical_and.reduce(mask)


def allocate_person_to_dataset(metas: pd.DataFrame) -> pd.DataFrame:
    """Function for allocating person to dataset

    Parameters
    ----------
    metas: pd.DataFrame
        meta data of HASC dataset.

    Returns
    -------
    metas: pd.DataFrame
        meta data of HASC dataset with person id and group assigned.
    """

    # {person: id, ...}
    people = dict(map(reversed, enumerate(metas['person'].unique())))
    # [[id, ...], ...]
    groups = np.array_split(
        np.random.permutation(list(map(int, people.values()))), 
        5,
    )
    # {group_id: [person_id, ...], ...}
    groups = dict(enumerate(groups))
    # {person_id: group_id, ...}
    groups = {person: group_id for group_id, group in groups.items() for person in group}

    metas['person_id'] = metas['person'].map(people).astype(int)
    metas['group_id'] = metas['person_id'].map(groups)
    return metas


def load_sensor_data(path: Path, meta: pd.DataFrame) -> Tuple[List[pd.DataFrame], pd.DataFrame]:

    def read_acc(path: Path) -> pd.DataFrame:
        if path.exists():
            try:
                ret = pd.read_csv(str(path), index_col=0, names=('x', 'y', 'z'))
                return ret
            except:
                print('[load] different data format:', str(path))
        else:
            print('[load] not found:', str(path))
        return pd.DataFrame()

    path = path / 'BasicActivity'
    data = [
        read_acc(path / row.act / row.person / '{}-acc.csv'.format(row.file))
        for row in meta.itertuples()
    ]

    return data, meta


def create_training_dataset(data: List[pd.DataFrame], meta: pd.DataFrame, output_path: Path):
    window_size = 256
    stride = 256
    front_trim = 100
    tail_trim = 100

    output_path.mkdir(exist_ok=True)

    labels = defaultdict(list)
    windows = list()
    for i, sensor_data in enumerate(data):
        act = meta.iloc[i]['act_id']
        person = meta.iloc[i]['person_id']

        sensor_data = sensor_data.values.copy()
        sensor_data = sensor_data[front_trim:-tail_trim].copy()  # (n, 3)
        view = np.lib.stride_tricks.sliding_window_view(sensor_data, window_shape=100, axis=0)  # (n - ws, 3, ws)
        view = view[::stride].copy()  # (n // stride, 3, ws)

        windows.append(view)
        labels['act'].extend([act] * len(view))
        labels['person'].extend([person] * len(view))
        labels['sensor'].extend([i] * len(view))
        labels['window'].extend(range(len(view)))

    windows = np.concatenate(windows, axis=0)   # (n, 3, ws)
    labels = dict(labels)
    labels = pd.DataFrame(labels)

    np.save(output_path / 'windows.npy', windows)
    labels.to_csv(output_path / 'labels.csv', index=False)


def main():
    Path('metas').mkdir(exist_ok=True)

    meta_path = Path('metas/hasc_meta.csv')
    if not meta_path.exists():
        path = Path('data/HASC-PAC2016_20160912/HASC-PAC2016')
        meta = load_meta(path)
        mask = mask_meta(meta)
        meta = meta[mask].copy()
        meta = allocate_person_to_dataset(meta)
        meta.to_csv('metas/metas.csv', index=False)
    meta = pd.read_csv('metas/metas.csv')

    for i in meta['group_id'].unique():
        group_meta = meta[meta['group_id'] == i].copy()

        data, group_meta = load_sensor_data(path, group_meta)
        output_path = Path('data/hasc_group{}'.format(i))
        output_path.mkdir(exist_ok=True)
        create_training_dataset(data, group_meta, output_path)


if __name__ == '__main__':
    main()
