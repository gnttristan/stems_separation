import base64
import io
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, STEMS, audio, compute_stft
from temporal_model import load_model, predict_frames


def predict(track, model_path):
    features = compute_stft(track)['amplitudes'].astype(np.float32)
    model, stems = load_model(model_path)
    if model.layers[0].in_channels != 200:
        raise ValueError('Rerun steps 4 and 5 to train the CNN on 200 amplitudes.')
    probabilities = predict_frames(features, model)
    return dict(probabilities=probabilities, labels=(probabilities >= .5).astype(np.uint8),
                seconds=np.arange(len(features)) * .01, stems=stems)


def expected_labels(track, frames):
    path = ROOT / '03_annotation/output' / f'{track.stem}.npy'
    if not path.exists():
        return None, 'No matching annotation for this filename.'
    labels = np.load(path)
    if labels.shape != (frames, len(STEMS)) or not np.isin(labels, [0, 1]).all():
        return None, 'Annotation does not match this audio; rerun step 3 with the original music.'
    return labels.tolist(), ''


def player_html(track, result):
    expected, message = expected_labels(track, len(result['labels']))
    wav = io.BytesIO()
    samples = (np.clip(audio(track), -1, 1) * 32767).astype(np.int16)
    wavfile.write(wav, 44100, samples)
    html = Path(__file__).with_name('player.html').read_text()
    return (html.replace('__AUDIO__', base64.b64encode(wav.getvalue()).decode())
            .replace('__LABELS__', json.dumps(result['labels'].tolist()))
            .replace('__STEMS__', json.dumps(result['stems'].tolist()))
            .replace('__EXPECTED__', json.dumps(expected))
            .replace('__EXPECTED_STEMS__', json.dumps(STEMS))
            .replace('__EXPECTED_MESSAGE__', json.dumps(message)))
