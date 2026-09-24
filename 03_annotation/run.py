import numpy as np
from common import ROOT, STEMS, folders, link


def collect_tracks(source):
    for track in sorted((ROOT / '02_preprocessing/output').glob('*/vocals.npz')):
        link(track.parent, source / track.parent.name)
    tracks = sorted(source.glob('*/vocals.npz'))
    if not tracks:
        raise ValueError('Run preprocessing first.')
    return tracks


def annotate(track, threshold=0.002):
    columns = []
    for stem in STEMS:
        with np.load(track / f'{stem}.npz') as data:
            columns.append(np.mean(data['amplitudes'], axis=1) > threshold)
    return np.column_stack(columns).astype(np.uint8)


def main():
    source, output = folders(__file__)
    for track in collect_tracks(source):
        np.save(output / f'{track.parent.name}.npy', annotate(track.parent))


if __name__ == '__main__':
    main()
