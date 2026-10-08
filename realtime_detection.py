"""
Real-time age & gender detection from a webcam.

Haar Cascade finds faces, the custom multi-output CNN predicts age and gender,
and detection_utils merges duplicate boxes and smooths the numbers over time.

Usage:
    python realtime_detection.py                 # default camera, quiet
    python realtime_detection.py --camera 1      # a different webcam
    python realtime_detection.py --debug         # raw + smoothed readout
    python realtime_detection.py --no-smooth     # per-frame values, as they come
    python realtime_detection.py --no-nms        # show every raw Haar box
    python realtime_detection.py --crop-policy square   # re-frame boxes like the training crops

Press 'q' in the video window to quit.
"""

import argparse
import os

import cv2
import numpy as np

from detection_utils import (
    DEFAULT_CROP_SCALE,
    DEFAULT_CROP_Y_SHIFT,
    FaceTracker,
    describe_crop_policy,
    face_crops,
    format_label,
    non_max_suppression,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "age_gender_custom_cnn_v1.keras")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade", "haarcascade_frontalface_default.xml")

IMG_SIZE = 128           # the model input is 128x128x1 grayscale
WINDOW_NAME = "Age & Gender Detection"


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time age/gender detection on a webcam")
    parser.add_argument(
        "--camera", type=int, default=0,
        help="cv2.VideoCapture device index (use 1 if 0 is taken or returns black frames)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="print raw and smoothed gender score / age for every detected face on every frame",
    )
    parser.add_argument("--min-face", type=int, default=60, help="minimum face size, in pixels")
    parser.add_argument("--scale-factor", type=float, default=1.3, help="Haar detector scale factor")
    parser.add_argument(
        "--smooth", type=int, default=5,
        help="EMA window per face, in frames. 1 = no smoothing (default: 5)",
    )
    parser.add_argument(
        "--no-smooth", action="store_true",
        help="shorthand for --smooth 1: keep tracking and label hysteresis, drop the averaging",
    )
    parser.add_argument(
        "--gender-margin", type=float, default=0.05,
        help="hysteresis half-band around 0.5; a label only flips past 0.5+/-margin. "
             "0.0 reproduces a plain 0.5 threshold",
    )
    parser.add_argument(
        "--nms-iou", type=float, default=0.3,
        help="merge boxes above this IoU into one detection (default: 0.3)",
    )
    parser.add_argument("--no-nms", action="store_true", help="disable duplicate-box merging")
    parser.add_argument(
        "--backend", choices=("any", "dshow", "msmf"), default="any",
        help="VideoCapture API. Windows' default MSMF path re-delivers the same frame "
             "to a slower reader; 'dshow' often stops that at the source",
    )
    parser.add_argument(
        "--crop-policy", choices=("haar", "square"), default="haar",
        help="'haar' feeds the raw detectMultiScale box (historical behaviour, still the "
             "default). 'square' re-frames the box to match the square UTKFace crops the "
             "network was trained on. Run scripts/eval_pipeline.py --search before switching "
             "this over: the two numbers below are a hypothesis, not a measurement.",
    )
    parser.add_argument(
        "--crop-scale", type=float, default=DEFAULT_CROP_SCALE,
        help="side of the square crop as a multiple of max(box w, h) (default: %(default)s)",
    )
    parser.add_argument(
        "--crop-y-shift", type=float, default=DEFAULT_CROP_Y_SHIFT,
        help="vertical shift of the square crop as a fraction of box height; negative moves it "
             "up, i.e. includes more forehead (default: %(default)s)",
    )
    parser.add_argument(
        "--no-skip-dupes", action="store_true",
        help="run inference on every read, even when the camera handed back an identical frame",
    )
    return parser.parse_args()


def load_model_and_cascade():
    """Loads the CNN and the face detector. Paths resolve next to this file, not the cwd."""
    # Imported here rather than at module scope so that tests of the crop policy,
    # the frame gate and the CLI can run without TensorFlow installed. Nothing
    # else in this module needs it: inference takes `model` as an argument.
    from tensorflow.keras.models import load_model

    if not os.path.isfile(MODEL_PATH):
        raise SystemExit(
            f"Model file not found: {MODEL_PATH}\n"
            "Clone the repo with the models/ folder intact, or retrain with scripts/train.py."
        )

    model = load_model(MODEL_PATH)
    print("Model loaded successfully")

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"Failed to load Haar Cascade from {CASCADE_PATH}")

    return model, face_cascade


def _feed(model, batch):
    """
    Wrap the batch as {input_name: batch} when the model has one named input.

    Keras 3 emits "UserWarning: The structure of `inputs` doesn't match the
    expected structure" on every call that passes a bare array to a Functional
    model whose input layer is named -- including model.predict(), so this was
    noisy before too. During a webcam session it prints twice per new face
    count, burying real output. Feeding by name is bit-identical (verified
    delta 0.0 on gender and age) and silent.
    """
    try:
        if len(model.inputs) == 1:
            name = model.inputs[0].name.split(":")[0]
            if name:
                return {name: batch}
    except Exception:
        pass
    return batch


def make_crops(gray, faces, policy="haar", scale=DEFAULT_CROP_SCALE, y_shift=DEFAULT_CROP_Y_SHIFT):
    """
    Turns detected boxes into a (N, IMG_SIZE, IMG_SIZE, 1) float32 batch.

    Split from the forward pass so an offline evaluation can stack crops from
    many images into one large batch and still run the exact function the webcam
    uses. That split is also what makes --search affordable: detection runs once
    per image and every candidate crop policy only re-crops.

    The crops carry RAW 0-255 pixels; the model's own Rescaling(1/255) layer
    normalizes them, and dividing here as well collapses every prediction to the
    dataset mean.
    """
    crops = [
        cv2.resize(c, (IMG_SIZE, IMG_SIZE)).astype("float32")
        for c in face_crops(gray, faces, policy=policy, scale=scale, y_shift=y_shift)
    ]
    if not crops:
        return np.zeros((0, IMG_SIZE, IMG_SIZE, 1), dtype="float32")
    return np.expand_dims(np.stack(crops), axis=-1)  # (N, 128, 128, 1)


def run_inference(model, crops):
    """
    One forward pass over a prepared crop batch, returning (gender_scores, ages).
    """
    if len(crops) == 0:
        return np.zeros((0,), dtype="float32"), np.zeros((0,), dtype="float32")

    try:
        # Direct call is much cheaper per frame than predict(), which rebuilds
        # its data pipeline machinery on every invocation.
        gender_pred, age_pred = model(_feed(model, crops), training=False)
        gender_pred, age_pred = gender_pred.numpy(), age_pred.numpy()
    except Exception:
        gender_pred, age_pred = model.predict(_feed(model, crops), verbose=0)

    return np.asarray(gender_pred).reshape(-1), np.asarray(age_pred).reshape(-1)


def predict_faces(model, gray, faces, policy="haar", scale=DEFAULT_CROP_SCALE,
                  y_shift=DEFAULT_CROP_Y_SHIFT):
    """
    Runs one batched forward pass for every face in the frame.

    Returns (gender_scores, ages) as 1-D float arrays aligned with `faces`.
    """
    return run_inference(model, make_crops(gray, faces, policy=policy, scale=scale, y_shift=y_shift))


_STRIDE = 8


class FrameGate:
    """
    Lets each distinct camera frame through once, absorbing re-reads.

    cv2.VideoCapture on Windows (MSMF) hands the same sensor sample back when
    the consumer runs slower than the sensor -- measured at 44% of reads in a
    live session, and 0% from the same camera under --backend dshow, which is
    where the problem actually belongs. Passing those through again spends
    inference on nothing new and double-weights the tracker's EMA on exactly
    the frames where the face is most stable, which is where you least want it
    biased. This gate is the safety net; dshow is the fix.
    """

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self.accepted = 0
        self.repeated = 0
        self._last = None

    def accept(self, frame):
        """True if this frame is new and should be analysed."""
        if not self.enabled:
            self.accepted += 1
            return True

        sig = frame_signature(frame)
        if self._last is not None and sig.shape == self._last.shape and np.array_equal(sig, self._last):
            self.repeated += 1
            return False

        self._last = sig
        self.accepted += 1
        return True

    def summary(self):
        total = self.accepted + self.repeated
        if not total:
            return "no frames read"
        return f"{self.accepted} analysed, {self.repeated} duplicate read(s) skipped ({100 * self.repeated / total:.0f}%)"


def frame_signature(frame):
    """
    Cheap fingerprint of a captured frame: every 8th pixel of one channel.

    Used only to spot a byte-identical re-read of the SAME camera frame, which
    the Windows MSMF capture path does routinely when the consumer is slower
    than the sensor. Sampling one channel every 8th pixel makes a false "same
    frame" verdict possible if something moved by less than 8 pixels, and the
    cost of that is one skipped inference pass on a scene that was already
    analysed a frame ago -- far cheaper than double-weighting the EMA on every
    static moment. Contiguous, so it does not pin the full frame buffer.
    """
    return np.ascontiguousarray(frame[::_STRIDE, ::_STRIDE, 0])


def analyze_frame(model, face_cascade, tracker, frame, args):
    """
    Detect -> merge -> predict -> smooth, for one frame.

    Drawn in place on `frame`; returns the list of row dicts it drew so a
    --debug caller (or a test) can inspect the numbers without a GUI.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = face_cascade.detectMultiScale(
        gray,
        scaleFactor=args.scale_factor,
        minNeighbors=5,
        minSize=(args.min_face, args.min_face),
    )
    faces = found if args.no_nms else non_max_suppression(found, iou_threshold=args.nms_iou)
    if len(faces) == 0:
        return []

    gender_scores, ages = predict_faces(
        model, gray, faces,
        policy=args.crop_policy, scale=args.crop_scale, y_shift=args.crop_y_shift,
    )
    rows = tracker.update(faces, gender_scores, ages)

    for row in rows:
        x, y, w, h = row["box"]
        label = format_label(row)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Console I/O is synchronous on Windows and will stall the loop, so
        # this is opt-in rather than per-face-per-frame by default.
        if args.debug:
            print(
                f"[face {row['id']}] gender {row['raw_gender_score']:.4f} -> {row['gender_score']:.4f}"
                f" | age {row['raw_age']:.2f} -> {row['age']:.2f} | {label}",
                flush=True,
            )

    return rows


def main():
    args = parse_args()
    if args.no_smooth:
        args.smooth = 1
    args.skip_dupes = not args.no_skip_dupes

    model, face_cascade = load_model_and_cascade()
    tracker = FaceTracker(window=args.smooth, gender_margin=args.gender_margin)

    backend = {"any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}[args.backend]
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera index {args.camera} with backend '{args.backend}'. "
            f"Try --camera 1, or another backend ({', '.join(b for b in ('any', 'dshow', 'msmf') if b != args.backend)}), "
            "and close anything holding the webcam (Windows Camera app, Teams, Zoom)."
        )

    try:
        # Fail here with a clear message rather than 60 lines into the loop.
        try:
            cv2.namedWindow(WINDOW_NAME)
        except cv2.error:
            raise SystemExit(
                "This OpenCV build has no GUI support (headless). "
                "Use scripts/predict_image.py --save instead, or install libgl1."
            )

        print(
            f"Smoothing window: {tracker.window} frame(s) | gender hysteresis: +/-{tracker.gender_margin:.2f}"
            f" | duplicate-box merging: {'off' if args.no_nms else f'IoU {args.nms_iou:.2f}'}"
        )
        print(
            f"            Crop policy: {describe_crop_policy(args.crop_policy, args.crop_scale, args.crop_y_shift)}"
        )

        gate = FrameGate(enabled=args.skip_dupes)
        processed_any = False

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera stopped returning frames; exiting.")
                break

            if gate.accept(frame):
                analyze_frame(model, face_cascade, tracker, frame, args)

            processed_any = True
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Unconditional: "0 duplicate read(s)" is the result of the dshow
        # experiment, not an absence of one, and silence cannot tell the two apart.
        if processed_any:
            print(f"Frames: {gate.summary()}."
                  + ("" if not gate.repeated else
                     " --backend dshow usually stops these at the source; "
                     "--no-skip-dupes re-enables per-read inference."))
    finally:
        # Always runs -- on Ctrl+C, on an exception, on 'q'. Skipping this leaves
        # the capture device locked on Windows, which looks like a broken camera
        # on the next run.
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
