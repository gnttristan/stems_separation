import numpy as np
import torch
from torch import nn

WINDOW = 31  # 310 ms at the existing 10 ms hop; predicts the center frame.


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class TemporalCNN(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(features, 32, 5), nn.ReLU(),
            nn.Conv1d(32, 16, 5, dilation=2), nn.ReLU(),
            nn.Conv1d(16, 4, 7, dilation=3),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)  # Four logits for BCEWithLogitsLoss.


def sequence_data(songs):
    """Pad songs individually; map each real frame to its center in padded data."""
    padded, centers, offset = [], [], 0
    for song in songs:
        padded.append(np.pad(song, ((15, 15), (0, 0)), mode='edge'))
        centers.append(np.arange(len(song)) + offset + 15)
        offset += len(song) + 30
    return np.concatenate(padded).astype(np.float32), np.concatenate(centers)


def batch_windows(data, indices, target=None):
    features, centers = data
    windows = features[centers[indices, None] + np.arange(-15, 16)]
    return torch.from_numpy(windows.transpose(0, 2, 1).copy()).to(target or device())


def save_model(path, model, stems):
    arrays = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    np.savez(path, **arrays, features=model.layers[0].in_channels, window=WINDOW,
             architecture='temporal_cnn_small', stems=stems)


def load_model(path, target=None):
    with np.load(path) as saved:
        if 'architecture' not in saved or str(saved['architecture']) != 'temporal_cnn_small':
            raise ValueError('Rerun step 5 to train the smaller temporal CNN.')
        model = TemporalCNN(int(saved['features']))
        model.load_state_dict({key: torch.from_numpy(saved[key]) for key in model.state_dict()})
        stems = saved['stems'].copy()
    return model.to(target or device()).eval(), stems


@torch.no_grad()
def predict_frames(features, model):
    data = sequence_data([features])
    model.eval()
    return np.concatenate([
        model(batch_windows(data, np.arange(start, min(start + 128, len(features))), next(model.parameters()).device)).sigmoid().cpu().numpy()
        for start in range(0, len(features), 128)
    ])
