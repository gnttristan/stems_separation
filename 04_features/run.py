import numpy as np
from common import ROOT, folders, link
from features import (get_abs_amp_features, get_mb_entropy, get_mb_onset,
                      get_mb_s_centroid, get_mb_s_flux, get_mb_zcr,
                      get_mb_crest_factor, normalize)


def extract(data):
    a, f = data['amplitudes'], data['frequencies']
    kwargs = dict(split_size=.05, power=1.)
    return np.vstack((*get_abs_amp_features(a, f), get_mb_entropy(a, **kwargs),
                      get_mb_onset(a, **kwargs), get_mb_s_centroid(a, f, **kwargs),
                      get_mb_s_flux(a, **kwargs), get_mb_zcr(a, **kwargs),
                      normalize(data['energy'].T, axis=1),
                      get_mb_crest_factor(a, **kwargs))).T.astype(np.float32)


def main():
    source, output = folders(__file__)
    for track in sorted((ROOT / '02_preprocessing/output').glob('*/full.npz')):
        link(track, source / f'{track.parent.name}.npz')
    tracks = sorted(source.glob('*.npz'))
    if not tracks:
        raise ValueError('Run preprocessing first.')
    for track in tracks:
        with np.load(track) as data:
            np.save(output / f'{track.stem}.npy', data['amplitudes'].astype(np.float32))


if __name__ == '__main__':
    main()
