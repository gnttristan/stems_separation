import json
import subprocess
import sys
import numpy as np
import pandas as pd
from common import ROOT, STEMS, folders, link, compute_stft


def collect_tracks(source):
    manifest = ROOT / '01_download/output/tracks.json'
    if manifest.exists():
        for name in json.loads(manifest.read_text()):
            link(manifest.parent / name, source / name)
    tracks = sorted(source.glob('*.wav'))
    if not tracks:
        raise ValueError('No WAV files in 02_preprocessing/input; run step 1 first.')
    return tracks


def preprocess_track(track, output):
    separated = output / 'audio/htdemucs' / track.stem
    if not all((separated / f'{stem}.wav').exists() for stem in STEMS):
        subprocess.run([sys.executable, '-m', 'demucs', '-n', 'htdemucs', '--float32',
                        '-o', str(output / 'audio'), str(track)], check=True)
    for stem in ('full', *STEMS):
        data = compute_stft(track if stem == 'full' else separated / f'{stem}.wav')
        target = output / track.stem
        target.mkdir(exist_ok=True)
        np.savez_compressed(target / f'{stem}.npz', **data)



def save_dataframes(tracks, output):
    for stem in ('full', *STEMS):
        columns = {}
        for track in tracks:
            with np.load(output / track.stem / f'{stem}.npz') as data:
                columns[track.stem] = dict(data)
        pd.DataFrame(columns).to_pickle(output / f'{stem}.pkl')
        del columns


def main():
    source, output = folders(__file__)
    tracks = collect_tracks(source)
    for track in tracks:
        preprocess_track(track, output)
    save_dataframes(tracks, output)


if __name__ == '__main__':
    main()
