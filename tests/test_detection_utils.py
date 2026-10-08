"""
Regression tests for the face-detection post-processing.

    python tests/test_detection_utils.py

No test framework required -- run it directly and it prints a count and exits 1
on failure. The functions are also named test_*, so `pytest tests/` collects them
if you have pytest.

Skips are deliberate, not laziness: some tests need OpenCV (crop geometry) and a
few need TensorFlow plus the 61 MB checkpoint (the parity numbers). A test whose
dependency is missing returns "skip" and the runner says so, instead of failing an
environment that legitimately has no model in it. Run it before and after changing
anything in detection_utils.py or a retrain, and treat the parity test as the
contract that the defaults have not silently moved.
"""

import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from detection_utils import (  # noqa: E402
    CROP_HAAR,
    CROP_POLICIES,
    CROP_SQUARE,
    DEFAULT_CROP_SCALE,
    DEFAULT_CROP_Y_SHIFT,
    FaceTracker,
    crop_window,
    describe_crop_policy,
    face_crop,
    face_crops,
    format_label,
    iou,
    non_max_suppression,
)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import realtime_detection as app
except ImportError:  # pragma: no cover - only if OpenCV itself is missing
    app = None

MODEL_PATH = os.path.join(BASE_DIR, "models", "age_gender_custom_cnn_v1.keras")
SAMPLE_IMAGE = os.path.join(BASE_DIR, "output", "output2.png")

CHECKS = {"n": 0, "failed": []}


def check(cond, msg):
    CHECKS["n"] += 1
    if not cond:
        CHECKS["failed"].append(msg)


def _gray(h=200, w=200):
    """Deterministic 8-bit image with a vertical ramp, so geometry is visible in values."""
    return np.tile(np.arange(h, dtype="uint8")[:, None], (1, w))


# ---------------------------------------------------------------------------
# non_max_suppression
# ---------------------------------------------------------------------------

def test_nms_shapes_and_types():
    out = non_max_suppression([])
    check(out.shape == (0, 4), f"empty NMS shape {out.shape}")
    check(out.dtype == np.int32, f"empty NMS dtype {out.dtype}")
    check(non_max_suppression(None).shape == (0, 4), "None input must be tolerated")

    single = non_max_suppression([(10, 10, 20, 20)])
    check(single.shape == (1, 4), f"single box shape {single.shape}")
    check(single.dtype == np.int32, "single box dtype")
    check(tuple(single[0]) == (10, 10, 20, 20), f"single box passthrough {single[0]}")


def test_nms_merges_shifted_duplicates():
    raw = np.array([[100, 100, 50, 50], [104, 103, 50, 50]], dtype="int32")
    out = non_max_suppression(raw)
    check(len(out) == 1, f"shifted duplicates should merge, got {len(out)}")
    check(tuple(out[0]) == (100, 100, 50, 50), f"the first/largest box should survive: {out[0]}")


def test_nms_keeps_separate_faces():
    raw = [[0, 0, 40, 40], [300, 300, 40, 40]]
    out = non_max_suppression(raw)
    check(len(out) == 2, f"two distant faces must survive, got {len(out)}")


def test_nms_drops_nested_box_by_containment():
    # Small box centred inside a big one: IoU 400/2500 = 0.16 (below 0.3) but
    # containment 1.0, which is the case IoU alone misses.
    raw = [[0, 0, 50, 50], [20, 20, 20, 20]]
    out = non_max_suppression(raw, iou_threshold=0.3, containment_threshold=0.7)
    check(len(out) == 1, f"nested box must be dropped by containment, got {len(out)}")
    loose = non_max_suppression(raw, iou_threshold=0.3, containment_threshold=1.01)
    check(len(loose) == 2, "with containment disabled the nested box should survive")


def test_nms_threshold_is_strict():
    # IoU exactly 1/3 for two 10x10 boxes offset by 5px.
    boxes = [[0, 0, 10, 10], [5, 0, 10, 10]]
    check(abs(iou(boxes[0], boxes[1]) - 1.0 / 3.0) < 1e-9, "IoU of the offset pair")
    check(len(non_max_suppression(boxes, iou_threshold=1.0 / 3.0)) == 2, "at the threshold, keep both")
    check(len(non_max_suppression(boxes, iou_threshold=0.3)) == 1, "just above 0.3, merge")


def test_nms_is_not_transitive_and_keeps_descending_area():
    # Greedy, largest-first: A swallows B, C overlaps B but not A, so C survives.
    a = [0, 0, 100, 100]
    b = [5, 5, 90, 90]
    c = [80, 80, 60, 60]
    out = non_max_suppression([c, a, b])
    check(len(out) == 2, f"expected A and C, got {len(out)}")
    check({tuple(b) for b in out} == {tuple(a), tuple(c)}, f"A and C survive, B is merged away: {out.tolist()}")
    check([tuple(b) for b in out] == [tuple(c), tuple(a)],
          "output keeps the INPUT order, which is what keeps tracker rows aligned with faces")
    check(out.dtype == np.int32, "merge output dtype")


def test_nms_is_idempotent():
    raw = [[10, 10, 50, 50], [12, 11, 50, 50], [400, 400, 40, 40], [401, 399, 44, 44]]
    once = non_max_suppression(raw)
    twice = non_max_suppression(once)
    check(np.array_equal(once, twice), "NMS applied twice must not change anything")


def test_nms_rejects_junk():
    raw = [[0, 0, 0, 10], [5, 5, -3, 10], [7, 7, 10, 10], None, (1, 2)]
    out = non_max_suppression(raw)
    check(len(out) == 1 and tuple(out[0]) == (7, 7, 10, 10), f"zero/negative area and malformed rows dropped: {out}")


# ---------------------------------------------------------------------------
# crop policy
# ---------------------------------------------------------------------------

def test_crop_haar_is_the_raw_box_as_a_view():
    g = _gray()
    box = (30, 20, 50, 40)
    crop = face_crop(g, box, policy=CROP_HAAR)
    check(crop.shape == (40, 50), f"haar crop shape {crop.shape} (H,W not W,H)")
    check(np.shares_memory(crop, g), "haar policy must be a zero-copy view, not a copy")
    check(np.array_equal(crop, g[20:60, 30:80]), "haar crop must equal the historical slice")
    check(crop.dtype == g.dtype, "crop must not change dtype")


def test_crop_window_params_are_ignored_by_haar():
    g = _gray()
    a = face_crop(g, (10, 10, 40, 40), policy=CROP_HAAR, scale=3.0, y_shift=2.0)
    b = face_crop(g, (10, 10, 40, 40), policy=CROP_HAAR)
    check(np.array_equal(a, b), "scale/shift must not touch the haar policy")
    check(crop_window((10, 10, 40, 40), g.shape, CROP_HAAR) == (10, 10, 50, 50, 0, 0, 0, 0), "window for haar")


def test_crop_square_is_square_and_centred():
    g = _gray()
    # A tall, thin Haar box: the square policy must not inherit its aspect.
    crop = face_crop(g, (100, 40, 40, 80), policy=CROP_SQUARE, scale=1.0, y_shift=0.0)
    check(crop.shape == (80, 80), f"square side should be max(w,h)={80}, got {crop.shape}")
    check(crop.shape[0] == crop.shape[1], "square policy always returns a square")

    x0, y0, x1, y1, pl, pt, pr, pb = crop_window((100, 40, 40, 80), g.shape, CROP_SQUARE, 1.0, 0.0)
    check(x1 - x0 == 80 and y1 - y0 == 80, "window side equals the pad-adjusted crop")
    check((pl, pt, pr, pb) == (0, 0, 0, 0), f"no padding expected in the middle of an image, got {(pl, pt, pr, pb)}")
    check(abs((x0 + x1) / 2.0 - 120.0) <= 1.0, "horizontally centred on the box")


def test_crop_square_scales_and_shifts():
    g = _gray()
    box = (100, 100, 40, 40)
    s1 = crop_window(box, g.shape, CROP_SQUARE, 1.0, 0.0)
    s2 = crop_window(box, g.shape, CROP_SQUARE, 1.5, 0.0)
    check(round(s2[2] - s2[0]) == 60, f"scale 1.5 on a 40px box -> 60px side, got {s2[2] - s2[0]}")
    up = crop_window(box, g.shape, CROP_SQUARE, 1.0, -0.5)
    down = crop_window(box, g.shape, CROP_SQUARE, 1.0, 0.5)
    check(up[1] == s1[1] - 20 and down[1] == s1[1] + 20, f"y_shift moves by shift*h: {up[1]} {s1[1]} {down[1]}")


def test_crop_pads_at_the_border_instead_of_shrinking():
    g = _gray(100, 100)
    # Box flush with the top-left corner, widened and lifted off the image.
    side = round(60 * 1.3)
    crop = face_crop(g, (0, 0, 60, 60), policy=CROP_SQUARE, scale=1.3, y_shift=-0.2)
    check(crop.shape == (side, side), f"crop must stay {side}x{side} at a border, got {crop.shape}")
    x0, y0, x1, y1, pl, pt, pr, pb = crop_window((0, 0, 60, 60), g.shape, CROP_SQUARE, 1.3, -0.2)
    check(x0 == 0 and y0 == 0, f"window clamped into the image, got {(x0, y0)}")
    check(pl > 0 and pt > 0, f"padding reported on the off-image sides, got {(pl, pt, pr, pb)}")
    check((x1 - x0) + pl + pr == side, "in-bounds width plus both pads equals the side")
    check(np.array_equal(crop[:, 0], crop[:, 1]), "left pad replicates the edge column (BORDER_REPLICATE)")
    check(crop.dtype == g.dtype, "padded crop keeps dtype")


def test_crop_never_crashes_on_degenerate_boxes():
    g = _gray(50, 50)
    for box in [(0, 0, 0, 0), (49, 49, 10, 10), (500, 500, 20, 20), (-10, -10, 5, 5)]:
        try:
            crop = face_crop(g, box, policy=CROP_SQUARE, scale=1.2, y_shift=-0.1)
            ok = crop.ndim == 2 and crop.shape[0] == crop.shape[1] and crop.size > 0
        except Exception as exc:  # noqa: BLE001 - a crash is exactly what we test for
            ok = False
            print(f"    face_crop({box}) raised {exc!r}")
        check(ok, f"degenerate box {box} must still produce a non-empty square")


def test_face_crops_order_and_empty_input():
    g = _gray()
    boxes = [(10, 10, 20, 20), (100, 100, 30, 30), (60, 20, 25, 25)]
    crops = face_crops(g, boxes, policy=CROP_SQUARE, scale=1.0, y_shift=0.0)
    check(len(crops) == 3, "one crop per box")
    check([c.shape for c in crops] == [(20, 20), (30, 30), (25, 25)], "output order follows input order")
    check(face_crops(g, []) == [], "empty box list -> empty list")
    check(face_crops(g, None) == [], "None box list -> empty list")


def test_crop_policy_api_surface():
    check(CROP_POLICIES == (CROP_HAAR, CROP_SQUARE), f"policies {CROP_POLICIES}")
    check(0.9 < DEFAULT_CROP_SCALE < 1.6, "default scale is a mild widening")
    check(-0.3 < DEFAULT_CROP_Y_SHIFT < 0.0, "default shift lifts the window")
    check("haar" in describe_crop_policy(CROP_HAAR), "descriptor for the default policy")
    check("1.15" in describe_crop_policy(CROP_SQUARE), f"descriptor shows the scale: {describe_crop_policy(CROP_SQUARE)}")
    try:
        face_crop(_gray(), (0, 0, 5, 5), policy="bogus")
        check(False, "unknown policy must raise")
    except ValueError:
        check(True, "unknown policy raises ValueError")


# ---------------------------------------------------------------------------
# FaceTracker
# ---------------------------------------------------------------------------

def test_tracker_window_math():
    t = FaceTracker(window=5)
    check(abs(t.alpha - 2.0 / 6.0) < 1e-12, "alpha = 2/(window+1)")
    check(t.smoothing is True, "window 5 smooths")
    one = FaceTracker(window=1)
    check(one.alpha == 1.0, f"window=1 must be alpha=1.0, got {one.alpha}")
    check(one.smoothing is False, "window=1 is 'no smoothing'")
    check(FaceTracker(window=0).window == 1, "window clamped to >= 1")
    check(FaceTracker(window=-4).window == 1, "negative window clamped to 1")
    check(FaceTracker(gender_margin=-1).gender_margin == 0.0, "negative margin clamped to 0")


def test_tracker_window_one_is_raw():
    t = FaceTracker(window=1, gender_margin=0.0)
    rows = t.update([(0, 0, 10, 10)], [0.9], [42.5])
    check(rows[0]["age"] == 42.5 and rows[0]["gender_score"] == 0.9, "window=1 reports the raw values")
    rows = t.update([(0, 0, 10, 10)], [0.1], [8.0])
    check(rows[0]["age"] == 8.0, f"no residual carry-over with window=1, got {rows[0]['age']}")


def test_tracker_ema_matches_hand_computation():
    t = FaceTracker(window=5, gender_margin=0.0)
    box = [(0, 0, 20, 20)]
    a = t.alpha
    t.update(box, [0.5], [10.0])
    rows = t.update(box, [0.7], [20.0])
    check(abs(rows[0]["age"] - (a * 20.0 + (1 - a) * 10.0)) < 1e-9, "EMA step on age")
    check(abs(rows[0]["gender_score"] - (a * 0.7 + (1 - a) * 0.5)) < 1e-9, "EMA step on gender")
    check(rows[0]["raw_age"] == 20.0 and rows[0]["raw_gender_score"] == 0.7, "raw_* keys stay raw")


def test_tracker_hysteresis_band():
    t = FaceTracker(window=1, gender_margin=0.05)
    box = [(0, 0, 10, 10)]
    check(t.update(box, [0.50], [30])[0]["gender"] == "Uncertain", "0.50 with no history is Uncertain")
    check(t.update(box, [0.55], [30])[0]["gender"] == "Female", "0.55 crosses the band")
    check(t.update(box, [0.52], [30])[0]["gender"] == "Female", "0.52 must not flip back inside the band")
    check(t.update(box, [0.46], [30])[0]["gender"] == "Female", "0.46 is still inside the band")
    check(t.update(box, [0.44], [30])[0]["gender"] == "Male", "0.44 exits below the band")
    check(t.update(box, [0.46], [30])[0]["gender"] == "Male", "and then it holds Male")
    check(t.update(box, [0.54], [30])[0]["gender"] == "Male", "0.54 is not enough to flip back")


def test_tracker_margin_zero_is_a_plain_threshold():
    t = FaceTracker(window=1, gender_margin=0.0)
    box = [(0, 0, 10, 10)]
    check(t.update(box, [0.50], [30])[0]["gender"] == "Female", "0.5 -> Female at margin 0")
    check(t.update(box, [0.4999], [30])[0]["gender"] == "Male", "just below flips immediately")
    check(t.update(box, [0.5001], [30])[0]["gender"] == "Female", "and back immediately -- no band")
    never = FaceTracker(window=1, gender_margin=0.0).update(box, [0.5], [30])[0]["gender"]
    check(never == "Female", "no Uncertain state exists at margin 0")


def test_tracker_ids_are_stable_and_separate_faces_are_independent():
    t = FaceTracker(window=2, gender_margin=0.05)
    a, b = (0, 0, 30, 30), (200, 200, 30, 30)
    for _ in range(3):
        rows = t.update([a, b], [0.1, 0.9], [8.0, 78.0])
    check([r["id"] for r in rows] == [0, 1], f"two faces get two ids in order, got {[r['id'] for r in rows]}")
    check(abs(rows[0]["age"] - 8.0) < 0.3 and abs(rows[1]["age"] - 78.0) < 0.3,
          "per-face smoothing: a child and a grandparent must not average together")
    rows = t.update([a], [0.1], [8.0])
    check(rows[0]["id"] == 0, "the surviving face keeps its id")


def test_tracker_jitter_within_iou_keeps_identity():
    t = FaceTracker(window=1, match_iou=0.05, gender_margin=0.0)
    rows1 = t.update([(100, 100, 50, 50)], [0.2], [20])[0]
    rows2 = t.update([(105, 103, 50, 50)], [0.2], [21])[0]  # shifted, still overlaps
    check(rows1["id"] == rows2["id"] == 0, f"small box jitter must not create a track: {rows1['id']}, {rows2['id']}")
    rows3 = t.update([(500, 500, 50, 50)], [0.2], [22])[0]
    check(rows3["id"] == 1, "a box with no overlap is a new face")


def test_tracker_expires_and_does_not_haunt_the_next_person():
    t = FaceTracker(window=5, max_age=2, gender_margin=0.0)
    box = [(100, 100, 50, 50)]
    for _ in range(5):
        t.update(box, [0.9], [80.0])
    check(t.active_tracks == 1, "one live track")
    for _ in range(3):
        t.update([], [], [])
    check(t.active_tracks == 0, f"track dropped after max_age misses, still {t.active_tracks}")
    rows = t.update(box, [0.1], [10.0])[0]
    check(rows["age"] == 10.0, f"a new face in the same box must not inherit age 80, got {rows['age']}")
    check(rows["gender"] == "Male", "and must not inherit the Female label")


def test_tracker_reset():
    t = FaceTracker(window=1, gender_margin=0.0)
    t.update([(0, 0, 10, 10)], [0.9], [50.0])
    check(t.active_tracks == 1, "track before reset")
    t.reset()
    check(t.active_tracks == 0, "no tracks after reset")
    check(t.update([(0, 0, 10, 10)], [0.1], [10.0])[0]["id"] == 0, "ids restart after reset")


def test_tracker_row_contract():
    t = FaceTracker(window=3, gender_margin=0.05)
    row = t.update([np.array([5, 6, 7, 8], dtype="int32")], [0.62], [27.4])[0]
    for key in ("id", "box", "gender", "gender_score", "age", "raw_gender_score", "raw_age"):
        check(key in row, f"row dict must carry '{key}'")
    check(row["box"] == (5, 6, 7, 8), f"box normalised to ints, got {row['box']}")
    check(t.update([], [], []) == [], "no faces returns an empty list, not None")


def test_format_label_truncates():
    row = {"gender": "Male", "gender_score": 0.451, "age": 25.99}
    check(format_label(row) == "Male, 25 (0.45)", f"got {format_label(row)!r}")
    check(format_label({"gender": "Uncertain", "gender_score": 0.5, "age": -3.0}) == "Uncertain, 0 (0.50)",
          "negative ages clamp to 0")
    check(format_label({"gender": "Female", "gender_score": 1.0, "age": 101.9}) == "Female, 101 (1.00)",
          "int truncation, not rounding")


# ---------------------------------------------------------------------------
# crop -> model batch (needs OpenCV)
# ---------------------------------------------------------------------------

def test_make_crops_legacy_path_is_unchanged():
    if app is None:
        return "skip"
    g = _gray()
    box = (30, 20, 50, 40)
    crops = app.make_crops(g, [box])
    expected = cv2.resize(g[20:60, 30:80], (app.IMG_SIZE, app.IMG_SIZE)).astype("float32")
    check(crops.shape == (1, app.IMG_SIZE, app.IMG_SIZE, 1), f"batch shape {crops.shape}")
    check(crops.dtype == np.float32, "batch dtype float32")
    check(np.array_equal(crops[0, ..., 0], expected), "the default policy must be bit-identical to the old inline resize")
    check(crops.max() <= 255.0 and crops.max() > 1.0, "crops carry RAW 0-255, not normalised pixels")
    check(app.make_crops(g, []).shape == (0, app.IMG_SIZE, app.IMG_SIZE, 1), "empty detection list")


def test_make_crops_square_policy_differs_and_stays_square():
    if app is None:
        return "skip"
    g = _gray()
    box = (60, 40, 40, 90)  # deliberately non-square, as Haar returns
    a = app.make_crops(g, [box], policy=CROP_HAAR)[0, ..., 0]
    b = app.make_crops(g, [box], policy=CROP_SQUARE)[0, ..., 0]
    check(a.shape == b.shape, "both policies land on the model input size")
    check(not np.array_equal(a, b), "the square policy must actually change what the model sees")
    src = face_crop(g, box, policy=CROP_SQUARE, scale=1.0, y_shift=0.0)
    check(src.shape[0] == src.shape[1], "pre-resize crop is square, so no stretch is baked in")


def test_run_inference_empty_batch_is_not_a_crash():
    if app is None:
        return "skip"
    g, a = app.run_inference(None, np.zeros((0, 128, 128, 1), dtype="float32"))
    check(len(g) == 0 and len(a) == 0, "empty crop batch short-circuits before the model")


def test_frame_gate_absorbs_the_msmf_double_read():
    if app is None:
        return "skip"
    gate = app.FrameGate()
    frames = [np.full((64, 64, 3), i, dtype="uint8") for i in range(20)]
    doubled = [f for pair in zip(frames, frames) for f in pair]
    accepted = sum(1 for f in doubled if gate.accept(f))
    check(accepted == 20, f"20 distinct frames delivered twice -> 20 accepted, got {accepted}")
    check(gate.repeated == 20, f"20 repeats expected, got {gate.repeated}")
    summary = gate.summary()
    check("20 analysed" in summary and "20 duplicate" in summary and "50%" in summary,
          f"summary must report both sides of the split: {summary}")

    off = app.FrameGate(enabled=False)
    check(all(off.accept(f) for f in doubled), "--no-skip-dupes accepts every read")
    check(off.repeated == 0, "and counts nothing as repeated")

    shape_change = app.FrameGate()
    shape_change.accept(np.zeros((64, 64, 3), dtype="uint8"))
    check(shape_change.accept(np.zeros((32, 32, 3), dtype="uint8")), "a different shape is never 'the same frame'")

    # Sub-stride motion can be missed by design; a change larger than the stride
    # must not be.
    sig = app.FrameGate()
    f1 = np.zeros((100, 100, 3), dtype="uint8")
    f2 = f1.copy()
    f2[40, 40, :] = 255
    sig.accept(f1)
    check(sig.accept(f2), "a change at frame[40,40] must be seen")
    check(app.frame_signature(f1).shape == app.frame_signature(f2).shape, "signature shape is stable")
    check(app.frame_signature(f1).flags["C_CONTIGUOUS"], "signature must be contiguous, not a strided view")


def test_cli_flags_exist_and_default_to_historical_behaviour():
    if app is None:
        return "skip"
    saved = sys.argv
    try:
        sys.argv = ["realtime_detection.py"]
        args = app.parse_args()  # noqa: E401
    finally:
        sys.argv = saved
    check(args.crop_policy == CROP_HAAR, "crop policy default stays haar until the numbers say otherwise")
    check(args.crop_scale == DEFAULT_CROP_SCALE, "crop scale default")
    check(args.crop_y_shift == DEFAULT_CROP_Y_SHIFT, "crop y shift default")
    check(args.smooth == 5 and args.gender_margin == 0.05 and args.nms_iou == 0.3, "stability defaults unchanged")
    check(args.backend == "any", "capture backend default unchanged")
    check(all(hasattr(args, d) for d in ("min_face", "scale_factor", "no_nms", "no_smooth", "debug")),
          "flag surface intact")
    try:
        sys.argv = ["realtime_detection.py", "--crop-policy", "nonsense"]
        app.parse_args()
        check(False, "an unknown crop policy must be rejected by argparse")
    except SystemExit:
        check(True, "unknown crop policy rejected")
    finally:
        sys.argv = saved


# ---------------------------------------------------------------------------
# scoreboard internals (no model needed for these)
# ---------------------------------------------------------------------------

def _load_eval():
    scripts = os.path.join(BASE_DIR, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        import eval_pipeline
    except (ImportError, SystemExit):
        # SystemExit because the module itself refuses to load without the model.
        return None
    return eval_pipeline


def test_eval_metrics_are_averages_over_scored_items_only():
    ev = _load_eval()
    if ev is None:
        return "skip"
    items = [{"age": 10.0, "gender": 0.0}, {"age": 20.0, "gender": 1.0}, {"age": 30.0, "gender": 0.0}]
    ages = np.array([12.0, 24.0, np.nan])
    genders = np.array([0.1, 0.9, np.nan])
    m = ev.metrics(items, ages, genders)
    check(m["n"] == 2, f"NaN rows excluded, got n={m['n']}")
    check(abs(m["mae"] - 3.0) < 1e-9, f"MAE over the two scored rows should be 3.0, got {m['mae']}")
    check(abs(m["gender_acc"] - 1.0) < 1e-9, "both genders correct")
    check(m["over_10y"] == 0.0, "no large errors here")
    empty = ev.metrics(items, np.full(3, np.nan), np.full(3, np.nan))
    check(empty is None, "all-NaN input must return None rather than divide by zero")
    bad = ev.metrics([{"age": 5.0, "gender": 0.0}], np.array([5.0]), np.array([0.9]))
    check(abs(bad["gender_acc"] - 0.0) < 1e-9 and not np.isfinite(bad["mae"] - 0.0) is False,
          "a below-chance gender accuracy is reported as-is; the caller warns")


def test_eval_metrics_flags_an_inverted_gender_mapping():
    ev = _load_eval()
    if ev is None:
        return "skip"
    items = [{"age": 30.0, "gender": g} for g in (0.0, 1.0, 0.0, 1.0)]
    ages = np.array([30.0] * 4)
    inverted = np.array([0.9, 0.1, 0.9, 0.1])
    m = ev.metrics(items, ages, inverted)
    check(abs(m["gender_acc"]) < 1e-9, "fully inverted scores read as 0% accuracy")
    out = []
    import io
    import contextlib

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ev.print_arm("x", m)
    except Exception as exc:  # noqa: BLE001
        out.append(str(exc))
    check(not out, f"print_arm must survive a below-chance arm: {out}")
    del buf


# ---------------------------------------------------------------------------
# parity against the historical numbers (needs TensorFlow + the checkpoint)
# ---------------------------------------------------------------------------

def test_parity_with_legacy_numbers():
    """
    output/output2.png through the default configuration must still land on the
    numbers this project has always printed. If a crop policy or a smoothing
    change moves these, that is a real behaviour change and it should be a
    deliberate one, written into the commit message, not a surprise.
    """
    if app is None or not os.path.isfile(MODEL_PATH) or not os.path.isfile(SAMPLE_IMAGE):
        return "skip"
    import importlib.util

    if importlib.util.find_spec("tensorflow") is None:
        return "skip"

    model, cascade = app.load_model_and_cascade()
    frame = cv2.imread(SAMPLE_IMAGE)
    if frame is None:
        return "skip"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
    faces = non_max_suppression(found)
    check(len(faces) >= 1, f"the sample image must still produce a detection, got {len(faces)}")

    genders, ages = app.predict_faces(model, gray, faces, policy=CROP_HAAR)
    check(abs(float(ages[0]) - 64.6654) < 0.01, f"legacy age 64.6654, got {float(ages[0]):.4f}")
    check(abs(float(genders[0]) - 0.60243) < 0.001, f"legacy gender 0.60243, got {float(genders[0]):.5f}")

    # The same image through the split functions must agree with predict_faces
    # exactly, or the eval script is measuring a different app than the one
    # people run.
    again_g, again_a = app.run_inference(model, app.make_crops(gray, faces, policy=CROP_HAAR))
    check(np.array_equal(again_a, ages) and np.array_equal(again_g, genders),
          "make_crops + run_inference must equal predict_faces")

    square_g, square_a = app.predict_faces(model, gray, faces, policy=CROP_SQUARE)
    check(abs(float(square_a[0]) - float(ages[0])) > 1e-4,
          "the square policy must produce different numbers, i.e. it really re-frames")


def test_analyze_frame_wires_the_crop_policy_through():
    """
    analyze_frame reads args attributes that argparse creates, so a renamed or
    missing flag is an AttributeError deep inside the camera loop. This drives the
    real function with real parsed args and a real model on the sample image.
    """
    if app is None or cv2 is None or not os.path.isfile(MODEL_PATH) or not os.path.isfile(SAMPLE_IMAGE):
        return "skip"
    import importlib.util

    if importlib.util.find_spec("tensorflow") is None:
        return "skip"

    saved = sys.argv
    try:
        sys.argv = ["realtime_detection.py"]
        args = app.parse_args()
    finally:
        sys.argv = saved

    model, cascade = app.load_model_and_cascade()
    frame = cv2.imread(SAMPLE_IMAGE)
    tracker = FaceTracker(window=5, gender_margin=0.05)
    rows = app.analyze_frame(model, cascade, tracker, frame, args)
    check(len(rows) == 1, f"the sample image should yield exactly one merged face, got {len(rows)}")
    if rows:
        check(abs(rows[0]["raw_age"] - 64.6654) < 0.01, "analyze_frame still reaches the model correctly")
        check(rows[0]["gender"] in ("Male", "Female", "Uncertain"), "label from the tracker, not a raw score")
    check(tracker.active_tracks == 1, "one track live after a single-face frame")

    # Same frame, different policy: the box is untouched, the numbers must move.
    sys.argv = ["realtime_detection.py", "--crop-policy", "square"]
    try:
        args2 = app.parse_args()
    finally:
        sys.argv = saved
    tracker.reset()
    rows2 = app.analyze_frame(model, cascade, tracker, frame, args2)
    if rows and rows2:
        check(abs(rows2[0]["raw_age"] - rows[0]["raw_age"]) > 1e-4,
              "the policy flag must actually reach the model input, not just parse")


def test_log_pattern_regressions():
    """
    Two failure modes this whole exercise started from, pinned down: identical
    raw values arriving twice per frame (the capture path), and a label flip
    caused by a score sitting next to 0.5.
    """
    if app is None:
        return "skip"
    gate = app.FrameGate()
    frame = np.random.default_rng(5).integers(0, 255, (80, 80, 3), dtype="uint8")
    seen = [gate.accept(frame) for _ in range(6)]
    check(seen == [True, False, False, False, False, False], f"the same frame must be analysed once, got {seen}")

    t = FaceTracker(window=5, gender_margin=0.05)
    box = [(0, 0, 40, 40)]
    labels = set()
    for score in [0.30, 0.32, 0.31, 0.42, 0.33, 0.31]:  # one spike toward the band
        labels.add(t.update(box, [score], [25.0])[0]["gender"])
    check(labels == {"Male"}, f"a single spike must not flip the label, saw {labels}")


# ---------------------------------------------------------------------------
# the scoreboard, end to end, with a fake model and a fake detector
# ---------------------------------------------------------------------------

class _FakeTensor:
    def __init__(self, arr):
        self._a = np.asarray(arr, dtype="float32")

    def numpy(self):
        return self._a


class _FakeModel:
    """
    Deterministic stand-in whose output depends on the crop, so a change to the
    crop policy has to show up in the numbers -- which is the property the real
    eval relies on and the thing a constant-output fake would hide.
    """

    def __call__(self, batch, training=False):
        arr = batch[list(batch)[0]] if isinstance(batch, dict) else batch
        means = arr.reshape(len(arr), -1).mean(axis=1) / 255.0
        return _FakeTensor(0.5 + 0.4 * means), _FakeTensor(30.0 + means)


class _FakeCascade:
    """Detects one box in every image except those whose marker pixel is a multiple of 4."""

    def detectMultiScale(self, gray, **kwargs):
        if int(gray[0, 0]) % 4 == 0:
            return np.zeros((0, 4), dtype="int32")
        h, w = gray.shape[:2]
        return np.array([[20, 20, w // 2, h // 2]], dtype="int32")


def test_eval_pipeline_end_to_end(tmp=None):
    """
    Runs eval_pipeline.main() over a synthetic folder with no dataset and no
    TensorFlow, so the wiring -- detection once, both arms, NaN handling for
    undetected images, the report, the search table, the strips, the JSON -- is
    covered by something other than a human reading a terminal.
    """
    import contextlib
    import io
    import json as _json
    import shutil
    import tempfile

    ev = _load_eval()
    if ev is None or cv2 is None:
        return "skip"

    work = tempfile.mkdtemp(prefix="evalpipe-")
    saved_argv, saved_load, saved_mlc = sys.argv, ev.load_test_split, ev.load_model_and_cascade
    try:
        items = []
        for i in range(24):
            img = np.tile((np.arange(120) * 2).astype("uint16").astype("uint8")[None, :], (120, 1))
            img = np.dstack([img] * 3)
            img[0, 0] = i  # marker the fake detector reads
            path = os.path.join(work, f"{10 + i}_0_0_2000010100000{i:02d}.png")
            cv2.imwrite(path, img)
            items.append({"path": path, "age": float(10 + i), "gender": 0.0})

        ev.load_test_split = lambda data_path, limit: items
        ev.load_model_and_cascade = lambda: (_FakeModel(), _FakeCascade())

        json_path = os.path.join(work, "res", "eval.json")
        strip_dir = os.path.join(work, "strips")
        sys.argv = [
            "eval_pipeline.py", "--data-path", work, "--search",
            "--scales", "1.0,1.2", "--shifts", "-0.1,0.0",
            "--dump-crops", "3", "--out-dir", strip_dir, "--json", json_path,
            "--crop-policy", "square", "--batch", "7",
        ]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ev.main()
        text = buf.getvalue()

        check(rc == 0, f"main() should exit 0, got {rc}")
        check("model arm" in text and "pipeline arm" in text, "both arms must be reported")
        check("face found in 18/24 images" in text, f"detector coverage miscounted:\n{text}")
        check("missed 6" in text, "undetected images must be counted, not silently dropped")
        check("detection + framing cost" in text, "the gap line is the headline number")
        check("4 square configs" in text, "grid should run 2x2 plus the haar baseline")
        check("WARNING" in text and "below chance" in text, "all-female predictions on all-male labels must warn")

        with open(json_path, encoding="utf-8") as fh:
            data = _json.load(fh)
        check(data["n_images"] == 24, f"json n_images {data.get('n_images')}")
        check(data["model_arm"]["n"] == 24, "the model arm scores every image, detection or not")
        check(data["pipeline_arm"]["n"] == 18, f"the pipeline arm scores only detected images: {data['pipeline_arm']}")
        check(abs(data["mae_gap"] - (data["pipeline_arm"]["mae"] - data["model_arm"]["mae"])) < 1e-9,
              "the reported gap must be the difference of the reported MAEs")
        check("policy" in data["search"]["best"] and "gain_vs_haar" in data["search"]["best"],
              f"search result must be machine-readable: {data['search']}")

        strips = sorted(f for f in os.listdir(strip_dir) if f.endswith(".png"))
        check(len(strips) == 3, f"expected 3 strips, got {strips}")
        if strips:
            strip = cv2.imread(os.path.join(strip_dir, strips[0]))
            check(strip is not None, "strip must be readable")
            check(strip.shape[1] == 3 * 128 * 3, f"three panels at 3x: {strip.shape}")
            check(strip.shape[0] == 3 * 128 + 26, f"caption row below the panels: {strip.shape}")

        # The number in the report has to be the number the arms produced; this
        # recomputes one arm independently and compares.
        ages, _ = ev.score_arm(_FakeModel(), items, "model", "haar", 1.0, 0.0, 7)
        expect = float(np.mean(np.abs(ages - np.array([it["age"] for it in items]))))
        check(abs(expect - data["model_arm"]["mae"]) < 1e-6,
              f"model-arm MAE {data['model_arm']['mae']} != recomputed {expect}")
    finally:
        sys.argv, ev.load_test_split, ev.load_model_and_cascade = saved_argv, saved_load, saved_mlc
        shutil.rmtree(work, ignore_errors=True)
    del _json, contextlib, io, shutil, tempfile


def test_eval_split_still_matches_training():
    """
    The two scripts must agree on what "the test split" means, so this always
    compares the two files' split calls structurally. When the training
    dependencies happen to be importable it additionally exercises the split
    itself, because that path is cheap here and expensive to get wrong.

    A silent divergence between the two splits is precisely the bug eval_pipeline
    was written to avoid -- it imports train.py for this one reason -- so the
    guard exists even in a half-installed environment where train.py cannot load.
    """
    ev = _load_eval()
    if ev is None:
        return "skip"
    import ast

    def split_calls(path):
        """
        train_test_split(...) calls as normalised source.

        Deliberately via the AST, not a regex over the text: a regex matched the
        split expression quoted inside eval_pipeline's own docstring and reported
        a drift that did not exist. A guard with that failure mode gets ignored
        within a week, which is worse than no guard.
        """
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                if name == "train_test_split":
                    found.append(ast.unparse(node))
        return sorted(found)

    src_calls = split_calls(os.path.join(BASE_DIR, "scripts", "eval_pipeline.py"))
    train_calls = split_calls(os.path.join(BASE_DIR, "scripts", "train.py"))
    check(len(src_calls) == 2, f"eval_pipeline should hold exactly two split calls, got {src_calls}")
    check(src_calls == train_calls,
          f"split expressions drifted:\n  eval: {src_calls}\n  train: {train_calls}")

    try:
        import train as train_mod  # noqa: F401
    except Exception:  # noqa: BLE001 - no TF/pandas in this environment
        return "skip"

    import pandas as pd

    df = pd.DataFrame({"filepath": [f"p{i}.jpg" for i in range(200)], "age": [float(10 + i % 40) for i in range(200)],
                       "gender": [float(i % 2) for i in range(200)]})
    tr1, tmp1 = train_mod.train_test_split(df, test_size=0.2, random_state=42)
    va1, te1 = train_mod.train_test_split(tmp1, test_size=0.5, random_state=42)
    tr2, tmp2 = train_mod.train_test_split(df, test_size=0.2, random_state=42)
    va2, te2 = train_mod.train_test_split(tmp2, test_size=0.5, random_state=42)
    check(te1["filepath"].tolist() == te2["filepath"].tolist(), "the split must be reproducible across calls")
    check(len(te1) == 20, f"10% of 200 images, got {len(te1)}")
    del pd


def main():
    tests = sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v))
    skipped = 0
    print(f"running {len(tests)} tests from {os.path.relpath(__file__, BASE_DIR)}")
    if cv2 is None:
        print("  (opencv not installed: crop/CLI tests will skip)")
    if not os.path.isfile(MODEL_PATH):
        print("  (no checkpoint: parity tests will skip)")
    print()
    for name, fn in tests:
        before = CHECKS["n"]
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - a crash is a failure, and it must be reported
            CHECKS["n"] += 1
            CHECKS["failed"].append(f"{name} raised {type(exc).__name__}: {exc}")
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
            continue
        if result == "skip":
            skipped += 1
            print(f"  skip  {name}")
        else:
            print(f"  ok    {name}  ({CHECKS['n'] - before} checks)")
    print()
    if CHECKS["failed"]:
        print(f"{len(CHECKS['failed'])} FAILED of {CHECKS['n']} checks:")
        for msg in CHECKS["failed"]:
            print(f"  - {msg}")
        return 1
    print(f"{CHECKS['n']} checks passed across {len(tests) - skipped} tests ({skipped} skipped).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
