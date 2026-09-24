# Stem activity pipeline

Use Python 3.11 and FFmpeg on PATH. Install with `python -m pip install -r requirements.txt`.

Edit `01_download/input/config.json` (SoundCloud likes URL, count, duration limit, optional browser for cookies). The default profile is inferred from the Led project.

```sh
python main.py all     # steps 1–5: download, separate, annotate, extract, train
python main.py 4       # rerun one step
# Put a new audio file in 06_inference/input, then:
python main.py 6
```

Each step has `input/` and `output/`. Inputs from earlier steps are symlinks to actual data, avoiding duplicate audio. Outputs appear when that step runs; no fabricated music or trained model is included.

| Step | Input | Output |
|---|---|---|
| 01_download | config.json | up to 100 eligible latest likes as WAV, tracks.json |
| 02_preprocessing | WAV files | Demucs audio; per-track full/vocals/drums/bass/other NPZ; five pandas pickles |
| 03_annotation | separated spectra | one uint8 `(frames, 4)` NPY per track |
| 04_features | full spectra and stereo energy | one float32 `(frames, 144)` NPY per track |
| 05_model | features and labels | model.npz, temporal CNN: 144 → 32 → 16 → 4 channels |
| 06_inference | model and new audio | NPZ with probabilities, binary labels, seconds, stem names |

STFT: 44.1 kHz, 25 ms Hann windows, 10 ms hop, 200 logarithmic bands. Mono amplitudes average channel magnitudes as in `stems_recognition`; stereo FFT energy is retained only for the mid/side feature. Frames are not grouped or dropped, preserving label alignment. Each dataframe has track IDs as columns and frequencies/amplitudes/energy as array-valued rows. Different track lengths need no padding. Load only trusted pickle files.

`process.md` lists overlapping maximum/largest-magnitude features and inconsistent arithmetic. The copied `stems_recognition/features.py` supplies four global features plus seven features for each of 20 bands (`split_size=.05`): **144 values**, not 110. Formulas match that file, with `power=1`; normalization is per song and feature to [-1, 1], constants map to zero. Inference processes a complete song with the same normalization. No clustering or extra flattened amplitude features are included.

Labels use exactly `any(amplitudes != 0)`, in vocals/drums/bass/other order. Demucs residuals may make every stem active almost everywhere; the trainer prints class proportions and flags constant labels. Magnitude sign-change features are zero by definition. Training concatenates all songs and shuffles aligned feature/label rows with seed 42, holding out 20% of frames (at least two frames required), prints binary cross-entropy and validation macro F1 each epoch, and saves the lowest-validation-loss model. `05_model/output/metrics.json` records the frame counts and seed, epoch history, and per-stem precision/recall/F1 at threshold 0.5. Undefined metrics are reported as zero. The requested 20% test partition is called validation in the logs because it selects the best epoch. It contains frames from the same songs as training, so it does not measure generalization to unseen songs. `split.npz` saves the exact row indices into the concatenated songs, in the saved track order.

Downloading needs network access; Demucs downloads its pretrained weights on first use. The downloader skips tracks over six minutes, missing durations, and failed downloads, continuing until 100 successful tracks or the likes list ends. For a different dataset, use fresh input/output folders to avoid mixing old and new tracks.

The temporal CNN uses the original 144 features over 31 consecutive frames (310 ms). Three Conv1d layers use channels 32/16/4, kernels 5/5/7 and dilation 1/2/3, with ReLU between layers and sigmoid probabilities at inference. Each window predicts its center frame; song edges repeat the nearest frame and windows never cross songs. Context includes future frames. NumPy handles data; the existing PyTorch dependency handles convolutions and gradients. Training uses Adam (learning rate 0.001, weight decay 0.0001), BCEWithLogitsLoss, batches of 128 and 50 epochs. Rerun step 5; existing step-4 features remain compatible, but old dense checkpoints must be replaced.

Training BCE is accumulated during optimization, avoiding a second pass over the training set each epoch. Full validation still runs every epoch. The smaller CNN checkpoint requires retraining step 5.

## Inference player

Run `python main.py 6` (or `python 06_inference/run.py`) to open the Streamlit player. Upload music, click Analyze audio, then play or seek. Drums, Vocals, Bass and Others light up independently at probability ≥ 0.5. The player uses the current CNN `05_model/output/model.npz`; uploads and predictions are saved in step 6 input/output. For file-only inference use `python 06_inference/run.py --batch`. Install the updated requirements first.

The player shows Expected above Predicted, synchronized to the same audio frame. Expected labels load from `03_annotation/output/<music filename without extension>.npy` in vocals/drums/bass/other order. Keep the original filename and audio timing; missing or incompatible annotations show an unavailable message while predictions remain playable.

Current amplitude experiment: step 4 saves the full-song STFT amplitudes directly as float32 `(frames, 200)` arrays. Step 5 trains the temporal CNN `200 → 32 → 16 → 4` with 31-frame windows; inference uses the same raw amplitudes. No rolling averages or additional normalization are applied. Feature extraction and temporal-context functions remain available but unused in this path. Rerun steps 4 and 5 before inference.
