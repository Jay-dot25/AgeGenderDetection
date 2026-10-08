"""
Score the whole app, not just the network.

The accuracy quoted in the README comes from the notebook, which evaluates the
model on UTKFace's own pre-cropped, pre-aligned 200x200 images. The webcam app is
a larger system: Haar proposes a box, that box is cropped and resized, and only
then does the network see a face. Nothing in this repository has ever measured
that path, so the number printed above the video window has always described a
friendlier problem than the one the app actually solves.

This runs the app's own functions -- same cascade, same duplicate merging, same
make_crops(), same forward pass -- over the *same* test split the model was
validated on, and reports both arms side by side:

    model arm      whole image resized to 128, i.e. the framing it was trained on
    pipeline arm   detected, cropped, resized: exactly what a webcam feeds it

The gap is the cost of detection and framing. That gap is also the only honest way
to judge --crop-policy square: if re-framing does not move the pipeline arm, the
idea was wrong and it should be dropped rather than kept because it sounds right.

Usage:
    python scripts/eval_pipeline.py --data-path C:\\data\\UTKFace
    python scripts/eval_pipeline.py --data-path C:\\data\\UTKFace --limit 500
    python scripts/eval_pipeline.py --data-path C:\\data\\UTKFace --search
    python scripts/eval_pipeline.py --data-path C:\\data\\UTKFace --dump-crops 12
    python scripts/eval_pipeline.py --data-path C:\\data\\UTKFace --json out\\eval.json

Needs the dataset plus requirements-train.txt (pandas, scikit-learn). The test
split is imported from scripts/train.py instead of being reimplemented here: a
second copy of train_test_split(df, test_size=0.2, random_state=42) is exactly the
kind of thing that quietly stops matching the first one, and then two scripts are
measuring two different sets while both claim to be "the test set".
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (BASE_DIR, os.path.join(BASE_DIR, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from detection_utils import (
        CROP_HAAR,
        CROP_SQUARE,
        DEFAULT_CROP_SCALE,
        DEFAULT_CROP_Y_SHIFT,
        describe_crop_policy,
        non_max_suppression,
    )
    from realtime_detection import (
        IMG_SIZE,
        load_model_and_cascade,
        make_crops,
        run_inference,
    )
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise SystemExit(
        f"Could not import the app modules: {exc}\n"
        "Install the inference dependencies first:  pip install -r requirements.txt"
    )

# scripts/train.py is imported inside load_test_split(), not here: it pulls in
# TensorFlow, pandas and scikit-learn at module scope, and a script that dies on
# import cannot be tested. The two lines that matter -- the split -- are still
# taken from train.py rather than copied, so they cannot drift from it.


# The crop policy the app uses today, as one row of the grid so the two are
# always compared on the same images.
HAAR_BASELINE = "haar"


# These two take comma-separated numbers that usually start with a minus sign, and
# argparse reads a leading "-" as the start of another flag. --shifts -0.2,0.0 would
# then die with "expected one argument", which is a confusing way to reject correct
# input, so the two spellings are folded together before parsing.
_LIST_VALUES = ("--scales", "--shifts")


def normalise_list_flags(argv):
    """Rewrite ``--shifts -0.2,0.0`` as ``--shifts=-0.2,0.0`` so argparse accepts it."""
    argv = list(argv)
    for i in range(len(argv) - 1):
        if argv[i] in _LIST_VALUES and argv[i + 1].startswith("-"):
            argv[i] = f"{argv[i]}={argv[i + 1]}"
            argv[i + 1] = None
    return [a for a in argv if a is not None]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Measure detection + framing cost on the UTKFace test split")
    parser.add_argument("--data-path", required=True, help="folder of UTKFace images (same one used for training)")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="evaluate only the first N test images (0 = all). Bounds the run, and the RAM: "
             "decoded frames are kept so --search does not re-read them.",
    )
    parser.add_argument("--min-face", type=int, default=60, help="must match the app to be comparable (default: 60)")
    parser.add_argument("--scale-factor", type=float, default=1.3, help="Haar scale factor, as in the app")
    parser.add_argument("--nms-iou", type=float, default=0.3, help="duplicate-box merging, as in the app")
    parser.add_argument("--batch", type=int, default=256, help="crop batches per forward pass")
    parser.add_argument(
        "--crop-policy", choices=(HAAR_BASELINE, "square"), default=HAAR_BASELINE,
        help="policy to report in the single-run table (default: haar, i.e. today's behaviour)",
    )
    parser.add_argument("--crop-scale", type=float, default=DEFAULT_CROP_SCALE)
    parser.add_argument("--crop-y-shift", type=float, default=DEFAULT_CROP_Y_SHIFT)
    parser.add_argument(
        "--search", action="store_true",
        help="sweep --crop-scale x --crop-y-shift under the square policy and rank by MAE",
    )
    parser.add_argument(
        "--scales", default="1.00,1.10,1.20,1.30",
        help="comma-separated crop scales for --search (default: %(default)s)",
    )
    parser.add_argument(
        "--shifts", default="-0.20,-0.12,-0.04,0.04",
        help="comma-separated y shifts for --search; a leading minus is fine here "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--dump-crops", type=int, default=0, metavar="N",
        help="write N side-by-side PNG strips showing what the network actually sees",
    )
    parser.add_argument("--out-dir", default=os.path.join("output", "eval_crops"), help="destination for --dump-crops")
    parser.add_argument("--json", help="write the numeric results here for later comparison")
    return parser.parse_args(normalise_list_flags(sys.argv[1:] if argv is None else argv))


def load_test_split(data_path, limit):
    """The model's own test split, produced by train.py's exact code path."""
    try:
        from train import load_dataframe, train_test_split
    except Exception as exc:  # noqa: BLE001 - any of TF/pandas/sklearn can fail to import
        raise SystemExit(
            f"Could not import scripts/train.py ({type(exc).__name__}: {exc}).\n"
            "This script reuses its dataset loading and split instead of copying them, so "
            "it also needs the training deps:  pip install -r requirements-train.txt"
        )

    df = load_dataframe(data_path)
    if len(df) < 20:
        raise SystemExit(
            f"Only {len(df)} usable images in {data_path}. This script evaluates the *test* "
            "split of train.py's 80/10/10 division, so a handful of files leaves nothing to "
            "score and sklearn fails with an opaque 'resulting train set will be empty'. "
            "Point --data-path at the full extracted UTKFace folder."
        )
    try:
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    except ValueError as exc:
        raise SystemExit(f"Could not reproduce train.py's split on {len(df)} images: {exc}")
    del train_df, val_df

    if limit and limit < len(test_df):
        test_df = test_df.head(limit)

    items = [
        {"path": row.filepath, "age": float(row.age), "gender": float(row.gender)}
        for row in test_df.itertuples(index=False)
    ]
    print(
        f"Test split: {len(items)} images (of {len(df)} usable overall). "
        "Same split as training, via scripts/train.py."
    )
    return items


def detect_all(items, cascade, args):
    """
    One detection pass for every image.

    Detection is the expensive, policy-independent half, so it runs once and each
    candidate crop policy reuses it. That is what makes a 16-cell grid search cost
    a couple of minutes instead of sixteen full runs.
    """
    for i, item in enumerate(items):
        frame = cv2.imread(item["path"])
        if frame is None:
            item["error"] = "unreadable"
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = cascade.detectMultiScale(
            gray,
            scaleFactor=args.scale_factor,
            minNeighbors=5,
            minSize=(args.min_face, args.min_face),
        )
        merged = non_max_suppression(found, iou_threshold=args.nms_iou)
        item["gray"] = gray
        item["n_raw"] = int(len(found))
        item["boxes"] = [tuple(int(v) for v in b) for b in merged]
        if item["boxes"]:
            # The subject in these images is the largest face; a background
            # false-positive is usually smaller than the real one.
            item["best"] = max(range(len(item["boxes"])), key=lambda k: item["boxes"][k][2] * item["boxes"][k][3])

    usable = [it for it in items if "gray" in it]
    broken = len(items) - len(usable)
    if broken:
        print(f"Skipped {broken} unreadable image file(s).")
    return usable


def score_arm(model, items, source, policy, scale, y_shift, batch):
    """
    Predict every item in `items` and return (ages, gender_scores) aligned to it.

    source="model" feeds the whole image, reproducing the training framing.
    source="pipeline" feeds the detector's chosen box, i.e. the app.
    Entries the pipeline arm cannot score (no face found) come back as NaN.
    """
    rows, idxs = [], []
    for k, item in enumerate(items):
        if source == "model":
            h, w = item["gray"].shape[:2]
            box = (0, 0, w, h)
        else:
            if "best" not in item:
                continue
            box = item["boxes"][item["best"]]
        crops = make_crops(item["gray"], [box], policy=policy, scale=scale, y_shift=y_shift)
        rows.append(crops[0])
        idxs.append(k)

    out_age = np.full(len(items), np.nan, dtype="float64")
    out_gender = np.full(len(items), np.nan, dtype="float64")
    if not rows:
        return out_age, out_gender

    stack = np.stack(rows)
    ages, genders = [], []
    for start in range(0, len(stack), max(1, batch)):
        g, a = run_inference(model, stack[start : start + batch])
        ages.append(a)
        genders.append(g)
    ages = np.concatenate(ages)
    genders = np.concatenate(genders)

    out_age[idxs] = ages
    out_gender[idxs] = genders
    return out_age, out_gender


def metrics(items, ages, genders):
    """MAE / gender accuracy over exactly the images that have a prediction."""
    ok = np.isfinite(ages) & np.isfinite(genders)
    if not ok.any():
        return None
    true_age = np.array([it["age"] for it in items], dtype="float64")[ok]
    true_gender = np.array([it["gender"] for it in items], dtype="float64")[ok]
    err = np.abs(ages[ok] - true_age)
    pred_female = genders[ok] >= 0.5
    acc = float((pred_female == (true_gender >= 0.5)).mean())
    return {
        "n": int(ok.sum()),
        "mae": float(err.mean()),
        "median_ae": float(np.median(err)),
        "p90_ae": float(np.percentile(err, 90)),
        "over_10y": float((err > 10).mean()),
        "gender_acc": acc,
    }


def print_arm(name, m):
    if m is None:
        print(f"{name:<34} no scorable images")
        return
    print(
        f"{name:<34} n={m['n']:<6} MAE {m['mae']:5.2f}y   median {m['median_ae']:5.2f}y   "
        f"p90 {m['p90_ae']:5.2f}y   >10y {m['over_10y'] * 100:4.1f}%   gender {m['gender_acc'] * 100:5.1f}%"
    )


def coverage_line(items):
    n = len(items)
    found = sum(1 for it in items if it.get("boxes"))
    multi = sum(1 for it in items if len(it.get("boxes", [])) > 1)
    raw = sum(it.get("n_raw", 0) for it in items)
    return (
        f"Detector: face found in {found}/{n} images ({100.0 * found / max(1, n):.1f}%), "
        f"missed {n - found}; >1 box after merging in {multi}; "
        f"{raw} raw boxes -> {sum(len(it.get('boxes', [])) for it in items)} after merging"
    )


def print_report(items, model_m, pipe_m, args, single_m=None, tag=""):
    print()
    print(f"--- {tag or 'results'} ---")
    print(f"Pipeline crop policy: {describe_crop_policy(args.crop_policy, args.crop_scale, args.crop_y_shift)}")
    print(coverage_line(items))
    print_arm("model arm (training framing)", model_m)
    print_arm(f"pipeline arm ({args.crop_policy})", pipe_m)
    if model_m and pipe_m:
        d_mae = pipe_m["mae"] - model_m["mae"]
        d_acc = pipe_m["gender_acc"] - model_m["gender_acc"]
        print(f"{'detection + framing cost':<34} +{d_mae:.2f}y MAE   {100 * d_acc:+.1f} pts gender")
        if model_m["n"] < 150:
            print(
                f"  (only {model_m['n']} images scored -- treat this gap as indicative, not as a "
                "measurement; drop --limit for a number worth quoting)"
            )
    if single_m:
        print(
            f"{'only where Haar found 1 box':<34} n={single_m['n']:<6} MAE {single_m['mae']:5.2f}y   "
            f"median {single_m['median_ae']:5.2f}y   gender {single_m['gender_acc'] * 100:5.1f}%   "
            "<- framing cost with detection noise removed"
        )
    for m in (model_m, pipe_m):
        if m and m["gender_acc"] < 0.5:
            print(
                f"\n  WARNING gender accuracy is below chance ({m['gender_acc'] * 100:.1f}%). "
                "The score is being read as P(Female) while the labels say otherwise, i.e. the "
                "mapping or the label convention is inverted somewhere. 1 - acc would be "
                f"{(1 - m['gender_acc']) * 100:.1f}%."
            )
            break


def search(model, items, args):
    """Grid over the square crop policy, ranked by MAE on the same images."""
    scales = [float(v) for v in args.scales.split(",") if v.strip()]
    shifts = [float(v) for v in args.shifts.split(",") if v.strip()]

    rows = []
    a, g = score_arm(model, items, "pipeline", CROP_HAAR, 1.0, 0.0, args.batch)
    base = metrics(items, a, g)
    if base:
        rows.append((base["mae"], "haar", 1.0, 0.0, base))
    print(f"\nbaseline {describe_crop_policy(CROP_HAAR)}: MAE {base['mae']:.3f}y" if base else "\nbaseline unavailable")

    t0 = time.time()
    for scale in scales:
        for shift in shifts:
            a, g = score_arm(model, items, "pipeline", CROP_SQUARE, scale, shift, args.batch)
            m = metrics(items, a, g)
            if m:
                rows.append((m["mae"], "square", scale, shift, m))
                print(f"  scale {scale:4.2f}  shift {shift:+5.2f}  ->  MAE {m['mae']:5.3f}y   "
                      f"gender {m['gender_acc'] * 100:5.1f}%")

    rows.sort(key=lambda r: r[0])
    print(
        f"\n--- crop search: {len(scales) * len(shifts)} square configs (+1 haar baseline) "
        f"in {time.time() - t0:.0f}s ---"
    )
    print(f"{'rank':<5}{'policy':<9}{'scale':>7}{'shift':>8}{'MAE':>8}{'gender':>9}{'vs haar':>10}")
    for rank, (mae, policy, scale, shift, m) in enumerate(rows[:10], start=1):
        delta = mae - base["mae"] if base else float("nan")
        print(f"{rank:<5}{policy:<9}{scale:7.2f}{shift:+8.2f}{mae:8.3f}{m['gender_acc'] * 100:8.1f}%{delta:+10.3f}")

    best_mae, best_policy, best_scale, best_shift, best_m = rows[0]
    gain = base["mae"] - best_mae if base else 0.0
    print()
    # A ranking over a handful of images moves by more than the differences it is
    # trying to read, and --search will happily crown a winner from noise.
    if base and base["n"] < 150:
        print(
            f"CAUTION: only {base['n']} images were scored. On a sample this small the spread "
            "between grid cells is mostly noise -- the 'best' config above could be a coin flip. "
            "Drop --limit and run the full test split (~2000 images) before acting on this table."
        )
    if gain <= 0.05:
        print(
            f"Best config beats the current crop by only {gain:+.3f}y MAE. That is not a real "
            "gain on this sample -- keep --crop-policy haar and do not ship the change. "
            "The framing hypothesis is not supported by your data."
        )
    else:
        print(f"Best config is {gain:+.3f}y better than the current crop:")
        print(f"  --crop-policy {best_policy} --crop-scale {best_scale} --crop-y-shift {best_shift}")
        print(
            "Before switching the default, re-run without --search to confirm the number holds, "
            "and eyeball --dump-crops 12 so you know the winner is winning for a sane reason "
            "(a crop can also win by cutting off the chin)."
        )
    return {
        "baseline_mae": base["mae"] if base else None,
        "best": {"policy": best_policy, "scale": best_scale, "y_shift": best_shift,
                 "mae": best_mae, "gender_acc": best_m["gender_acc"], "gain_vs_haar": gain},
    }


def dump_crops(model, items, args):
    """
    Write strips of what the network is actually fed, so framing is a screenshot
    rather than an argument.
    """
    try:
        os.makedirs(args.out_dir, exist_ok=True)
    except OSError as exc:
        print(f"--dump-crops: cannot create {args.out_dir}: {exc}")
        return
    wanted = max(0, int(args.dump_crops))
    written = 0
    for item in items:
        if written >= wanted:
            break
        if "best" not in item:
            continue
        h, w = item["gray"].shape[:2]
        box = item["boxes"][item["best"]]
        panels = [
            ("training framing", make_crops(item["gray"], [(0, 0, w, h)], policy=CROP_HAAR)[0]),
            (f"haar box {box}", make_crops(item["gray"], [box], policy=CROP_HAAR)[0]),
            ("square policy", make_crops(
                item["gray"], [box], policy=CROP_SQUARE,
                scale=args.crop_scale, y_shift=args.crop_y_shift,
            )[0]),
        ]
        views = []
        for title, crop in panels:
            img = cv2.cvtColor(np.clip(crop, 0, 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
            img = cv2.resize(img, (IMG_SIZE * 3, IMG_SIZE * 3), interpolation=cv2.INTER_NEAREST)
            cv2.putText(img, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            views.append(img)
        strip = np.vstack([np.hstack(views), np.full((26, IMG_SIZE * 3 * 3, 3), 32, dtype="uint8")])
        truth = f"true  age {item['age']:.0f}  {'Female' if item['gender'] >= 0.5 else 'Male'}"
        cv2.putText(strip, truth, (6, IMG_SIZE * 9 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        name = os.path.splitext(os.path.basename(item["path"]))[0]
        dest = os.path.join(args.out_dir, f"{written:02d}_{name}.png")
        try:
            if not cv2.imwrite(dest, strip):
                print(f"cv2.imwrite returned False for {dest}")
                break
        except cv2.error as exc:
            print(f"cv2.imwrite refused {dest}: {str(exc).splitlines()[0]}")
            break
        written += 1
    if written < wanted:
        print(f"Only {written} of {wanted} requested strips: the rest had no detection to crop.")
    print(f"Wrote {written} crop strip(s) to {os.path.abspath(args.out_dir)}")


def main():
    args = parse_args()
    if not os.path.isdir(args.data_path):
        raise SystemExit(
            f"No such folder: {args.data_path}\n"
            "Point --data-path at the extracted UTKFace directory (the one containing "
            "files named like 24_1_0_20000101005723.jpg)."
        )

    items = load_test_split(args.data_path, args.limit)
    if not items:
        raise SystemExit("The test split is empty -- nothing to evaluate.")

    model, cascade = load_model_and_cascade()
    t0 = time.time()
    items = detect_all(items, cascade, args)
    print(f"Detected in {len(items)} images in {time.time() - t0:.0f}s")
    if not items:
        raise SystemExit("No image could be read; nothing to score.")

    model_age, model_gender = score_arm(model, items, "model", CROP_HAAR, 1.0, 0.0, args.batch)
    pipe_age, pipe_gender = score_arm(
        model, items, "pipeline",
        CROP_HAAR if args.crop_policy == HAAR_BASELINE else CROP_SQUARE,
        args.crop_scale, args.crop_y_shift, args.batch,
    )
    model_m = metrics(items, model_age, model_gender)
    pipe_m = metrics(items, pipe_age, pipe_gender)

    # The same images the detector agreed on exactly once: the gap here is pure
    # framing, with no false-positive or double-box contamination in it.
    single = [i for i, it in enumerate(items) if len(it.get("boxes", [])) == 1]
    single_m = None
    if len(single) >= 10:
        single_m = metrics(
            [items[i] for i in single],
            np.array([pipe_age[i] for i in single]),
            np.array([pipe_gender[i] for i in single]),
        )

    print_report(items, model_m, pipe_m, args, single_m=single_m)

    results = {
        "n_images": len(items),
        "crop_policy": describe_crop_policy(args.crop_policy, args.crop_scale, args.crop_y_shift),
        "detector": {"min_face": args.min_face, "scale_factor": args.scale_factor, "nms_iou": args.nms_iou},
        "model_arm": model_m,
        "pipeline_arm": pipe_m,
        "mae_gap": (pipe_m["mae"] - model_m["mae"]) if (model_m and pipe_m) else None,
    }

    if args.search:
        results["search"] = search(model, items, args)

    if args.dump_crops:
        dump_crops(model, items, args)

    if args.json:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2, sort_keys=True)
            print(f"Wrote {os.path.abspath(args.json)}")
        except OSError as exc:
            print(f"--json: could not write {args.json}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
