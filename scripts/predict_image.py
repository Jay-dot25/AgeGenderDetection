"""
Run age/gender prediction on a single image file (no webcam needed).

This is also the headless path for servers and containers: pass --save and
nothing ever tries to open a GUI window.

Usage:
    python scripts/predict_image.py --image path/to/photo.jpg
    python scripts/predict_image.py --image path/to/photo.jpg --save out/annotated.jpg
"""

import argparse
import os
import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "age_gender_custom_cnn_v1.keras")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade", "haarcascade_frontalface_default.xml")

# detection_utils.py lives at the repo root, one level above scripts/.
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from detection_utils import (  # noqa: E402
    DEFAULT_CROP_SCALE,
    DEFAULT_CROP_Y_SHIFT,
    FaceTracker,
    face_crops,
    format_label,
    non_max_suppression,
)

IMG_SIZE = 128           # the model input is 128x128x1 grayscale


def parse_args():
    parser = argparse.ArgumentParser(description="Age/gender prediction on a single image")
    parser.add_argument("--image", required=True, help="Path to an input image")
    parser.add_argument("--save", help="Optional path to save the annotated output image")
    parser.add_argument("--min-face", type=int, default=60, help="minimum face size, in pixels")
    parser.add_argument("--scale-factor", type=float, default=1.3, help="Haar detector scale factor")
    parser.add_argument(
        "--nms-iou", type=float, default=0.3,
        help="merge boxes above this IoU into one detection (default: 0.3)",
    )
    parser.add_argument("--no-nms", action="store_true", help="disable duplicate-box merging")
    parser.add_argument(
        "--gender-margin", type=float, default=0.05,
        help="report 'Uncertain' while the score sits within this band of 0.5 instead of "
             "picking a side; 0.0 reproduces a plain 0.5 threshold",
    )
    return parser.parse_args()


def predict_faces(model, gray, faces):
    """
    One batched forward pass for every face found. Returns (gender_scores, ages)
    aligned with `faces`.
    """
    # Same crop helper as the webcam path, so the two cannot drift apart and
    # scripts/eval_pipeline.py measures a framing that this tool also uses.
    crops = [
        cv2.resize(c, (IMG_SIZE, IMG_SIZE)).astype("float32")
        for c in face_crops(
            gray, faces, policy="haar",
            scale=DEFAULT_CROP_SCALE, y_shift=DEFAULT_CROP_Y_SHIFT,
        )
    ]
    if not crops:
        return np.zeros((0,), dtype="float32"), np.zeros((0,), dtype="float32")
    crops = np.expand_dims(np.stack(crops), axis=-1)  # (N, 128, 128, 1)

    # Feed RAW 0-255 pixels -- the model has its own Rescaling(1/255) layer built in.
    gender_pred, age_pred = model.predict(_feed(model, crops), verbose=0)

    return np.asarray(gender_pred).reshape(-1), np.asarray(age_pred).reshape(-1)


def _feed(model, batch):
    """Wrap the batch as {input_name: batch} for a single named input.

    Keras 3 warns ("The structure of `inputs` doesn't match the expected
    structure") on every bare-array call, including model.predict(). Feeding by
    name is silent and bit-identical.
    """
    try:
        if len(model.inputs) == 1:
            name = model.inputs[0].name.split(":")[0]
            if name:
                return {name: batch}
    except Exception:
        pass
    return batch


def save_annotated(frame, path):
    """
    Writes the image and fails loudly if OpenCV could not actually write it.

    imwrite has two distinct failure modes, both of which used to go unnoticed:
    it returns False for a missing parent directory, and it *raises* cv2.error
    for an extension it has no encoder for.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)  # imwrite does NOT create missing folders

    try:
        wrote = cv2.imwrite(path, frame)
    except cv2.error as e:
        raise SystemExit(f"cv2.imwrite could not write {path}: {e}")

    if not wrote:
        raise SystemExit(
            f"cv2.imwrite returned False for {path}. OpenCV only writes extensions it "
            "knows (.png, .jpg, .jpeg, .bmp, .tiff) and cannot write to a locked folder."
        )

    print(f"Annotated image saved to: {os.path.abspath(path)}")


def main():
    args = parse_args()

    if not os.path.isfile(MODEL_PATH):
        raise SystemExit(f"Model file not found: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"Failed to load Haar Cascade from {CASCADE_PATH}")

    frame = cv2.imread(args.image)
    if frame is None:
        raise IOError(f"Could not read image: {args.image}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = face_cascade.detectMultiScale(
        gray,
        scaleFactor=args.scale_factor,
        minNeighbors=5,
        minSize=(args.min_face, args.min_face),
    )
    faces = found if args.no_nms else non_max_suppression(found, iou_threshold=args.nms_iou)

    if len(faces) < len(found):
        print(f"Merged {len(found) - len(faces)} duplicate box(es): {len(found)} detections -> {len(faces)} faces")

    if len(faces) == 0:
        print("No faces detected.")
        if args.save:
            # Still emit the file so batch scripts get a consistent output set.
            save_annotated(frame, args.save)
        return 1

    gender_scores, ages = predict_faces(model, gray, faces)

    # window=1: a still image has no frames to average against, but routing the
    # score through FaceTracker keeps the label wording and the Uncertain band
    # identical to the live webcam path instead of drifting from it.
    tracker = FaceTracker(window=1, gender_margin=args.gender_margin)
    for row in tracker.update(faces, gender_scores, ages):
        label = format_label(row)
        print(label)

        x, y, w, h = row["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if args.save:
        save_annotated(frame, args.save)
    else:
        # The verdicts are already on stdout; the window is only a convenience, so
        # a headless OpenCV build must not turn a successful run into a traceback.
        try:
            cv2.imshow("Age & Gender Detection", frame)
            print("Showing the annotated image -- press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            reason = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(
                "This OpenCV build has no GUI support, so no window was shown "
                f"({reason}). The labels above are the result; pass --save PATH to "
                "write the annotated image instead."
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
