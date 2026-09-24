import json
import numpy as np
import torch
from torch import nn
from common import ROOT, STEMS, folders, link
from temporal_model import TemporalCNN, sequence_data, batch_windows, save_model, load_model


def collect_tracks(source):
    for path in sorted((ROOT / '04_features/output').glob('*.npy')):
        link(path, source / 'features' / path.name)
        label = ROOT / '03_annotation/output' / path.name
        if not label.exists():
            raise ValueError(f'Missing annotation: {label}')
        link(label, source / 'labels' / path.name)
    tracks = sorted((source / 'features').glob('*.npy'))
    if not tracks:
        raise ValueError('Run annotation and feature extraction first.')
    return tracks


def load_dataset(tracks, source):
    features, labels = [], []
    dimensions = 200
    for path in tracks:
        x, y = np.load(path, mmap_mode='r'), np.load(source / 'labels' / path.name)
        if x.ndim != 2 or x.shape[1] != dimensions or y.shape != (len(x), len(STEMS)):
            raise ValueError(f'Feature/label shape mismatch: {path.name}; rerun steps 3 and 4.')
        if not len(x) or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
            raise ValueError(f'Expected finite features and binary labels: {path.name}')
        features.append(x)
        labels.append(y)
    x, y = sequence_data(features), np.concatenate(labels)
    if len(y) < 2:
        raise ValueError('The frame split needs at least two frames.')
    return x, y


def split_frames(x, y, tracks, output, rng):
    print('Positive label fractions:', dict(zip(STEMS, y.mean(axis=0))), flush=True)
    indices = rng.permutation(len(y))
    cut = max(1, int(len(y) * .8))
    training, validation = indices[:cut], indices[cut:]
    np.savez(output / 'split.npz', training=training, validation=validation, tracks=[p.stem for p in tracks])
    print(f'Frame split: {len(training)} training, {len(validation)} validation (seed=42)', flush=True)
    return training, validation


@torch.no_grad()
def evaluate(x, y, model, indices):
    model.eval()
    loss, count = 0., 0
    tp, fp, fn, support = np.zeros((4, len(STEMS)))
    for start in range(0, len(indices), 128):
        batch = indices[start:start + 128]
        labels = y[batch]
        p = model(batch_windows(x, batch)).sigmoid().numpy().astype(float)
        predicted = p >= .5
        p = np.clip(p, 1e-7, 1 - 1e-7)
        loss -= np.sum(labels * np.log(p) + (1 - labels) * np.log1p(-p))
        count += labels.size
        tp += (predicted & (labels == 1)).sum(axis=0)
        fp += (predicted & (labels == 0)).sum(axis=0)
        fn += (~predicted & (labels == 1)).sum(axis=0)
        support += labels.sum(axis=0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return loss / count, precision, recall, f1, support


def train_epoch(x, y, model, training, rng, optimizer):
    model.train()
    total_loss = 0.
    shuffled = rng.permutation(training)
    for start in range(0, len(shuffled), 128):
        indices = shuffled[start:start + 128]
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_windows(x, indices))
        loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(y[indices]).float())
        total_loss += loss.item() * len(indices)
        loss.backward()
        optimizer.step()
    return total_loss / len(training)


def train(x, y, training, validation, output, rng, epochs=50):
    torch.manual_seed(42)
    model = TemporalCNN(x[0].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
    history, best_loss, best_epoch = [], np.inf, 0
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(x, y, model, training, rng, optimizer)
        val_loss, _, _, f1, _ = evaluate(x, y, model, validation)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_macro_f1=float(f1.mean())))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            save_model(output / 'model.npz', model, STEMS)
        print(f'Epoch {epoch:02d}/{epochs} | train BCE {train_loss:.4f} | val BCE {val_loss:.4f} | val macro F1 {f1.mean():.3f}', flush=True)
    return best_epoch, history


def report_metrics(x, y, weights, validation, best_epoch):
    _, precision, recall, f1, support = evaluate(x, y, weights, validation)
    print(f'Best model: epoch {best_epoch}; validation metrics at threshold 0.5:')
    metrics = {}
    for i, stem in enumerate(STEMS):
        metrics[stem] = dict(precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]), positive_frames=int(support[i]))
        print(f'{stem:6s} | precision {precision[i]:.3f} | recall {recall[i]:.3f} | F1 {f1[i]:.3f} | positive frames {int(support[i])}')
    return metrics


def main():
    source, output = folders(__file__)
    tracks = collect_tracks(source)
    x, y = load_dataset(tracks, source)
    rng = np.random.default_rng(42)
    training, validation = split_frames(x, y, tracks, output, rng)
    best_epoch, history = train(x, y, training, validation, output, rng)
    model, _ = load_model(output / 'model.npz')
    metrics = report_metrics(x, y, model, validation, best_epoch)
    report = dict(architecture='temporal_cnn_small', channels=[200, 32, 16, 4], input_kind='amplitudes', window_frames=31, split='random_frames', seed=42, training_frames=len(training), validation_frames=len(validation),
                  best_epoch=best_epoch, history=history, metrics=metrics)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(f'Saved {output / "model.npz"} and metrics.json')


if __name__ == '__main__':
    main()
