from pathlib import Path
import subprocess
import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parent
STEMS = ('vocals', 'drums', 'bass', 'other')


def folders(script):
    base = Path(script).resolve().parent
    return base / 'input', base / 'output'


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(source.resolve())


def audio(path):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path),
                                  '-ar', '44100', '-ac', '2', '-f', 'f32le', '-'])
    return np.frombuffer(raw, dtype='<f4').reshape(-1, 2)


def compute_stft(path):
    """SciPy STFT: 25 ms windows, 10 ms hops, no boundary padding."""
    samples = audio(path)
    n, hop = round(44100 * .025), 441
    if len(samples) < n:
        raise ValueError(f'Audio is shorter than 25 ms: {path}')
    window = signal.windows.hann(n, sym=True)
    frequencies, _, values = signal.stft(
        samples.astype(float).T, fs=44100, window=window,
        nperseg=n, noverlap=n - hop, boundary=None, padded=False,
    )
    values = values.transpose(2, 1, 0) * (2 * window.sum() / n)
    values[:, 0] = 0
    return frequency_bands(frequencies, values)


def frequency_bands(fft_frequencies, values):
    """Convert SciPy's bins to the 200 bands and stereo energy used by features."""
    frequencies = np.geomspace(20, 22050, 200)
    edges = np.r_[0, np.sqrt(frequencies[:-1] * frequencies[1:]), np.inf]
    bins = np.digitize(fft_frequencies, edges) - 1
    groups = np.searchsorted((200 * np.arange(.05, 1, .05)).astype(int), bins, side='right')
    magnitude = np.abs(values).mean(axis=2)
    mid_side = (np.abs((values[..., 0] + values[..., 1]) / 2) ** 2 +
                np.abs((values[..., 0] - values[..., 1]) / 2) ** 2) / 2
    amplitudes = np.array([magnitude[:, bins == band].sum(axis=1) for band in range(200)]).T
    energy = np.array([mid_side[:, groups == band].sum(axis=1) / max((groups == band).sum(), 1)
                       for band in range(20)]).T
    return dict(frequencies=frequencies, amplitudes=amplitudes, energy=energy)


def temporal_context(x, windows=(5, 15)):
    """Append centered 100/300 ms means; clip windows at this song's edges."""
    cumulative = np.vstack((np.zeros((1, x.shape[1])), np.cumsum(x, axis=0, dtype=float)))
    frames = np.arange(len(x))
    context = [x]
    for window in windows:
        start = np.maximum(frames - window // 2, 0)
        end = np.minimum(frames + (window + 1) // 2, len(x))
        context.append((cumulative[end] - cumulative[start]) / (end - start)[:, None])
    return np.hstack(context).astype(np.float32)


def forward(x, model):
    """Shared 64-ReLU → 32-ReLU → four-sigmoid forward pass."""
    if 'w3' not in model or 'w4' in model or 'w1' not in model or model['w1'].shape[0] != x.shape[1]:
        raise ValueError('Model is incompatible; rerun step 5 to train the temporal neural network.')
    h1 = np.maximum(x @ model['w1'] + model['b1'], 0)
    h2 = np.maximum(h1 @ model['w2'] + model['b2'], 0)
    logits = h2 @ model['w3'] + model['b3']
    p = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    return h1, h2, p


def probability(x, model):
    return forward(x, model)[-1]
