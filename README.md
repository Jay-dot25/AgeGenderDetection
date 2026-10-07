# Real-Time Age & Gender Detection
### Custom CNN (Keras) + Haar Cascade Face Detection — Trained on UTKFace

Detects faces from a webcam (or a single image) and predicts **age** and **gender** in real time using a custom-trained multi-output CNN. Fully local — no API calls, no internet required at inference time.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│            Webcam / Image Input              │
└──────────────────────┬───────────────────────┘
                        │ BGR frame
┌──────────────────────▼───────────────────────┐
│          Haar Cascade Face Detector           │
│   haarcascade/haarcascade_frontalface_default │
│              .xml                             │
└──────────────────────┬───────────────────────┘
                        │ face ROI (grayscale, NxM)
┌──────────────────────▼───────────────────────┐
│         Custom Multi-Output CNN (Keras)       │
│  Input: 128×128×1 grayscale                   │
│  Rescaling(1/255) ← baked into the model      │
│  4× [Conv2D → MaxPool]  (32→64→128→256)       │
│  Flatten (shared trunk)                       │
│  ├─ Dense(256) → Dropout → output_gender      │
│  └─ Dense(256) → Dropout → output_age         │
└──────────────────────┬───────────────────────┘
                        │ gender (sigmoid), age (linear)
┌──────────────────────▼───────────────────────┐
│        OpenCV Overlay (box + label)           │
└────────────────────────────────────────────────┘
```

*Between the model and the overlay sits a post-processing stage in `detection_utils.py` — duplicate-box merging and per-face smoothing. See [Prediction stability](#prediction-stability).*

## Training Pipeline

```
UTKFace dataset
  filenames: {age}_{gender}_{race}_{date}.jpg
    │
    ▼
Parse filename → age, gender labels
    │
    ▼
train / val / test split (80 / 10 / 10)
    │
    ▼
tf.data pipeline: decode jpeg → resize 128×128 → grayscale
    │
    ▼
Data Augmentation (flip, rotation, zoom)   [train only]
    │
    ▼
Rescaling(1/255)                            [inside the model]
    │
    ▼
Custom CNN — multi-output (gender + age)
    │
    ▼
models/age_gender_custom_cnn_v1.keras
```

## Model Details

| Output | Type | Activation | Loss |
|--------|------|------------|------|
| `output_gender` | Binary classification (0 = Male, 1 = Female) | sigmoid | binary_crossentropy |
| `output_age` | Regression (years) | linear | mae |

Input is always **128×128, single-channel grayscale**, raw pixel values in `[0, 255]`. Normalization happens *inside* the model via a `Rescaling(1./255.)` layer — see [Known Issues](#known-issues--lessons-learned) below for why this matters.

## Requirements

| | |
|---|---|
| Python | **3.9 – 3.12** (no TensorFlow 2.16 wheel exists for 3.13+) |
| TensorFlow | `>=2.16,<2.17` — 2.16 bundles **Keras 3**, and the shipped model was saved with Keras 3.8. TF 2.15 (Keras 2) **cannot load it** |
| numpy | `>=1.24,<2` — TF 2.16 rejects numpy 2.x |
| Needs a webcam? | Only `realtime_detection.py`. `scripts/predict_image.py` works headless |

## Setup (First Time)

### macOS / Linux

```bash
git clone <your-repo-url>
cd AgeGenderDetection

# Inference-only setup
bash setup.sh

# If you also want to retrain the model
bash setup.sh --train
```

### Windows (PowerShell)

`setup.sh` is a bash script — skip it and run the four commands it wraps:

```powershell
git clone <your-repo-url>
cd AgeGenderDetection

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes for Windows checkouts:

- Use `py -3.11` / `python`, **not** `python3` (that name is a dead Store alias), and `.\.venv\Scripts\Activate.ps1`, not `source .venv/bin/activate`.
- If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force`.
- Prefer a folder **outside OneDrive** — Files On-Demand keeps the 59 MB `models/*.keras` as a remote placeholder and it can read as missing.

## Running

```bash
# Real-time webcam detection
python realtime_detection.py
python realtime_detection.py --camera 1      # if index 0 is taken or shows black
python realtime_detection.py --debug         # raw -> smoothed readout per face
python realtime_detection.py --no-smooth     # per-frame values, as they come from the model
python realtime_detection.py --no-nms        # show every raw Haar box, duplicates included

# Single image (no webcam needed)
python scripts/predict_image.py --image path/to/photo.jpg --save output/annotated.jpg
```

| Flag | Default | Meaning |
|---|---|---|
| `--camera N` | `0` | `cv2.VideoCapture` index |
| `--smooth N` | `5` | EMA window per face, in frames (`1` = off) |
| `--no-smooth` | – | shorthand for `--smooth 1` |
| `--gender-margin X` | `0.05` | hysteresis half-band around 0.5; `0.0` = plain threshold |
| `--nms-iou X` | `0.3` | merge Haar boxes above this overlap |
| `--no-nms` | – | keep every box the detector returns |
| `--min-face N` | `60` | smallest face to look for, in px |
| `--scale-factor X` | `1.3` | detector step size; higher = faster, fewer misses turned detections |

Activate the venv first (`source .venv/bin/activate`, or `.\.venv\Scripts\Activate.ps1`), or call the venv interpreter directly and skip activation entirely:

```bash
.\.venv\Scripts\python.exe realtime_detection.py     # Windows
./.venv/bin/python realtime_detection.py              # macOS / Linux
```

Press `q` to quit the webcam window. All faces in a frame are batched into one forward pass, and the camera is always released on exit.

## Prediction stability

The CNN is fine; `cv2.CascadeClassifier` is not a tracker, and neither is a per-frame threshold. Three things in `detection_utils.py` close that gap, all measured against a real 239-frame webcam session:

**Duplicate boxes → `non_max_suppression()`.** `detectMultiScale` routinely reports the same face two or three times. In the recorded session **102 of 137 distinct predictions came in identical pairs**, which draws a doubled rectangle and ghost-doubles the label text, and pays 2x inference for nothing. Boxes are ranked by area (Haar gives no confidence score), then any later box that overlaps a kept one is dropped on *either* IoU `> 0.3` *or* containment `> 0.7` — the second test catches a small box nested inside a big one, whose IoU alone can look innocent (`0.09`). Two real adjacent faces score low on both and survive.

**Jittery numbers → `FaceTracker` EMA.** Frame-to-frame age moved by 1.71 years on average, worst case 13.32 years, with 57 jumps over 3 years. Each detected face keeps its own exponentially-weighted average (`alpha = 2/(window+1)`) of age and gender score, matched across frames by IoU so two people in one frame are never averaged together. On a noisy sequence this took the spread from sd 5.69 to sd 1.48 while still tracking a genuine drift, and a steady face is not drifted at all.

**Flapping labels → hysteresis.** **41% of frames sat in the 0.40–0.50 band** — right under a 0.5 cutoff, one pixel of noise from flipping the verdict. A face already labelled `Male` stays `Male` until the smoothed score clears `0.5 + margin` (and vice versa); before any label is established, a score inside the band reads `Uncertain` rather than guessing. That first `Uncertain` is a feature, not a regression: it means the model genuinely cannot tell.

`--no-smooth --gender-margin 0 --no-nms` reproduces the original behavior exactly (verified: same 64.6654 age and 0.60243 score on the sample image), so you can A/B the pipeline against the raw model output.

## Retraining the Model

The model was trained on [UTKFace](https://susanqq.github.io/UTKFace/). Download it, then either:

**Option A — Notebook** (includes EDA + plots; needs `bash setup.sh --train` first):
```bash
python -m notebook notebooks/ageandgender.ipynb
```

**Option B — Script:**
```bash
python scripts/train.py --data-path /path/to/UTKFace --epochs 50
```

Notes:

- Image discovery accepts `*.jpg`, `*.jpeg` **and** `*.png` (UTKFace mirrors differ), and it stops with a clear message instead of an opaque `shuffle(buffer_size=0)` error when the folder has nothing usable in it.
- Rows whose gender label is not `0` or `1` are dropped — some mirrors encode "unknown" as `2`, which corrupts a sigmoid head.
- The exported model has the training-only augmentation layers **stripped**, so the saved graph is pure inference. Feed it raw `0–255` pixels as before.
- `models/age_gender_custom_cnn_v1.keras` is overwritten by default; pass `--output models/my_run.keras` to keep the current one and diff the two.
- If you point `--data-path` at the extracted archive root instead of the inner `UTKFace/` folder, you'll get the "No images matched" error — that nesting is normal for this dataset.

## Project Structure

```
AgeGenderDetection/
├── realtime_detection.py      # Live webcam inference
├── detection_utils.py         # NMS + per-face smoothing/hysteresis (no model deps)
├── scripts/
│   ├── train.py               # Script version of the training notebook
│   └── predict_image.py       # Single-image inference (no webcam, works headless)
├── notebooks/
│   └── ageandgender.ipynb     # Full training notebook (EDA, training, eval plots)
├── models/
│   └── age_gender_custom_cnn_v1.keras   # 59MB, committed (see Big files below)
├── haarcascade/
│   └── haarcascade_frontalface_default.xml
├── output/                    # Sample annotated results (regenerable)
├── requirements.txt           # Inference-only deps
├── requirements-train.txt     # Extra deps for (re)training
├── setup.sh                   # bash only (macOS/Linux)
├── .gitattributes             # line-ending + binary rules
└── LICENSE
```

## Known Issues / Lessons Learned

**Double-normalization bug (fixed):** the model's first real layer is `Rescaling(1./255.)`, meaning it expects raw `0–255` pixel values as input and divides by 255 itself. An earlier version of `realtime_detection.py` *also* divided the face crop by 255 before feeding it to the model, so every input was effectively scaled down to `~0–0.004` — close enough to a blank frame that the network just output its learned average for every face (age stuck around 45–48, gender always low-confidence "Male", regardless of who or what was in front of the camera). The fix was simply to stop normalizing manually and let the model's own `Rescaling` layer do it, since that's exactly how `scripts/train.py` / the notebook feed images during training.

**Detector jitter (mitigated in software):** `cv2.CascadeClassifier` keeps no temporal state and returns no confidence score, so raw per-frame output double-counts faces and swings years of age between consecutive frames. `detection_utils.py` corrects this at inference rather than by retraining; measured before/after is in [Prediction stability](#prediction-stability).

**Dataset skew:** UTKFace is skewed toward adult faces; expect lower accuracy on children and the elderly unless you augment with a more balanced dataset.

**Augmentation layers inside the saved model (fixed for future runs):** `data_augmentation` (`RandomFlip`/`RandomRotation`/`RandomZoom`) used to be part of the graph that got serialized, so every inference run carried dead layers — inert at `training=False`, but dead weight in a 59MB artifact. `scripts/train.py` now rebuilds the same topology with `augment=False`, copies the weights across (the augmentation stack holds no weights, so `get_weights()` lines up 1:1) and saves *that*. The committed `age_gender_custom_cnn_v1.keras` still has the submodel; retraining replaces it.

**Keras 2 vs 3:** the committed model is a Keras 3 file. Under TF 2.15 (Keras 2.15), `load_model()` fails with:

```
TypeError: Could not deserialize class 'Functional' because its parent module
keras.src.models.functional cannot be imported.
```

That is why `requirements.txt` floors at 2.16. If you hit it, you have an older TensorFlow left over from another project — `python -m pip install --force-reinstall "tensorflow>=2.16,<2.17"`.

## Big files

`models/age_gender_custom_cnn_v1.keras` (59 MB) and the `output/` samples are committed as plain git objects, with no LFS. Clones are therefore slow and browser/OneDrive copies of the repo frequently end up **missing the big files** — if `requirements.txt` and the `.py` files are present but `models/` isn't, you have a partial copy and need a real `git clone`. To keep new weights out of history:

```bash
git lfs install
git lfs track "*.keras"      # then add .gitattributes' lfs rule
# or better: ship large artifacts as a GitHub Release asset and download on setup
```

Existing history isn't rewritten by this — that needs `git lfs migrate`, which rewrites every commit hash.

## Troubleshooting

**`ModuleNotFoundError: No module named 'tensorflow'`**
→ Activate the venv first (`source .venv/bin/activate` / `.\.venv\Scripts\Activate.ps1`), or run the venv interpreter by path.

**`TypeError: Could not deserialize class 'Functional'`**
→ Your env has TensorFlow 2.15 / Keras 2, which cannot read this Keras 3 model. `python -m pip install --force-reinstall "tensorflow>=2.16,<2.17"` — see [Known Issues](#known-issues--lessons-learned).

**`Could not open camera index 0`**
→ `python realtime_detection.py --camera 1`. On Windows the Camera app, Teams or Zoom holding the device produces this too — close them. If `--camera` guessing is tedious, probe the indexes:
```bash
python -c "import cv2; [print(i, cv2.VideoCapture(i).isOpened()) for i in range(3)]"
```

**Black window that never updates after the first run**
→ The camera was left locked because the old script exited without `cap.release()`. The loop now runs inside `try/finally`. If you're on an older copy, kill the stuck python process (Task Manager / `pkill -f realtime_detection`) or unplug/replug the webcam.

**`This OpenCV build has no GUI support (headless)` / `ImportError: libGL.so.1`**
→ Linux containers and WSL lack the GL libs. Either `pip install opencv-python-headless --force-reinstall` and use `scripts/predict_image.py --save`, or `sudo apt-get install -y libgl1 libglib2.0-0`.

**`Annotated image saved to: ...` but no file exists**
→ Fixed. `cv2.imwrite` has two failure modes: it returns `False` for a missing parent directory (the old code printed "saved" anyway) and it *raises* `cv2.error: could not find a writer for the specified extension` for a format OpenCV cannot encode. Both are now handled, and the parent folder is created for you. Valid extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`.

**Gender accuracy reads as a huge percentage (e.g. `3432.00%`) after retraining**
→ You're on the old notebook/train.py, which read `evaluate()` results by index. Keras 3 puts `output_age_mae` where Keras 2 puts `output_gender_accuracy`, so those two numbers swap and MAE gets printed as a "percentage". Fixed by reading results by key via `evaluate(return_dict=True)`. Don't try `model.metrics_names` as a workaround — under Keras 3 it returns a `compile_metrics` entry and omits the head metric names, so the list doesn't even line up with the results.

**`A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`**
→ `python -m pip install "numpy<2" --force-reinstall`, and install all of `requirements.txt` in one `pip install` pass so the resolver can pick 1.26.x for you.

**`Found 0 image files` when retraining**
→ Either the archive wasn't extracted (UTKFace nests images in an inner `UTKFace/` folder) or it's the `.png` variant, which the old glob ignored. Point `--data-path` at the folder that actually holds the images.

**Training run prints a metric list that doesn't match its own results**
→ Fixed. `model.metrics_names` does still exist on Keras 3.0 – 3.15 (verified), but under Keras 3 it returns `['loss', 'compile_metrics', 'output_gender_loss', 'output_age_loss']` for a model whose `evaluate()` returns **five** values — the two head metrics are missing and a `compile_metrics` placeholder is inserted, so the printed "names" line up with nothing. Worse, `test_results[3]`/`[4]` swap meaning between Keras 2 and Keras 3 (see above). Both `scripts/train.py` and notebook cell 26 now use `evaluate(return_dict=True)` and read by key, which is correct on either Keras.

**The same face gets two boxes and doubled-up text**
→ Duplicate Haar rectangles. The default NMS merge removes them; `--no-nms` reproduces them, and `--nms-iou 0.15` merges more aggressively if pairs still show.

**The age number bounces around between frames**
→ Expected from a frame-independent detector. `--smooth 5` is the default; go higher (`--smooth 12`) for a steadier number at the cost of a little lag when you genuinely move.

**The label reads `Uncertain`**
→ The gender score is inside the +/-0.05 band around 0.5, i.e. the model genuinely cannot tell on this face. `--gender-margin 0` forces a binary call like the original code, at the cost of flipping between frames.

**Predictions look constant / barely change across faces**
→ Check you're not normalizing pixel values before passing them to the model — see [Known Issues](#known-issues--lessons-learned).

**`Failed to load Haar Cascade`**
→ Make sure you cloned the repo with the `haarcascade/` folder intact; the path is resolved relative to the script, not your current working directory.

## License

MIT — see [LICENSE](LICENSE).
