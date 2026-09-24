import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description='Music player with live stem activity indicators.')
    parser.add_argument('--batch', action='store_true', help='Save predictions for files in input/ without opening the player.')
    args = parser.parse_args()
    if args.batch:
        import numpy as np
        from inference import ROOT, predict
        source, output = Path(__file__).parent / 'input', Path(__file__).parent / 'output'
        tracks = [p for p in source.iterdir() if p.suffix.lower() in ('.wav', '.mp3', '.flac', '.ogg', '.m4a')]
        if not tracks:
            raise ValueError('Put a music file in 06_inference/input, or run without --batch to upload one.')
        for track in tracks:
            np.savez_compressed(output / f'{track.stem}.npz', **predict(track, ROOT / '05_model/output/model_cnn.npz'))
    else:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', str(Path(__file__).with_name('app.py'))], check=True)


if __name__ == '__main__':
    main()
