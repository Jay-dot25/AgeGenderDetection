"""
Shared post-processing for the face-detection pipeline.

Used by realtime_detection.py (webcam) and scripts/predict_image.py (still
image). Deliberately free of any model or OpenCV dependency so the geometry
and smoothing logic can be reasoned about -- and tested -- on its own.

Two jobs, because cv2.CascadeClassifier gives you neither:

1. non_max_suppression() -- detectMultiScale returns overlapping rectangles for
   the SAME face (in a real session roughly three quarters of detections came in
   duplicate pairs), which doubles the boxes, ghost-doubles the label text and
   doubles inference cost for zero information.

2. FaceTracker -- Haar is frame-independent, so the raw predictions jitter:
   measured frame-to-frame age swings averaged 1.7 years with a 13-year
   maximum, and a gender score sitting near 0.5 flips the printed label between
   identical frames. Per-face EMA plus label hysteresis fixes both.
"""

import numpy as np


def _to_boxes(faces):
    """Normalise whatever detectMultiScale returned into a list of int (x, y, w, h)."""
    boxes = []
    for f in ([] if faces is None else faces):
        try:
            x, y, w, h = (int(v) for v in tuple(f)[:4])
        except (TypeError, ValueError):
            continue
        if w > 0 and h > 0:
            boxes.append((x, y, w, h))
    return boxes


def _geometry(a, b):
    """Returns (intersection_area, iou, containment) for two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy

    area_a, area_b = aw * ah, bw * bh
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    smaller = min(area_a, area_b)
    containment = inter / smaller if smaller > 0 else 0.0
    return inter, iou, containment


def iou(a, b):
    """Intersection over union of two (x, y, w, h) boxes."""
    return _geometry(a, b)[1]


def non_max_suppression(faces, iou_threshold=0.3, containment_threshold=0.7):
    """
    Merge rectangles that describe the same face.

    Haar boxes carry no confidence score to rank by, so they are kept in
    descending-area order (the outer box of a Haar group is the tighter fit for
    the face) and any later box that overlaps a kept one too much is dropped.
    Two tests, because the failure modes differ:

      * iou > iou_threshold          -- side-by-side shifted duplicates
      * containment > threshold      -- a small box nested inside a big one,
                                        whose IoU alone can look harmless

    Two genuinely separate faces score low on both, so they survive.

    Returns an (N, 4) int32 array, i.e. the same shape and dtype family as
    detectMultiScale's output, so call sites do not change.
    """
    boxes = _to_boxes(faces)
    if not boxes:
        return np.zeros((0, 4), dtype="int32")
    if len(boxes) == 1:
        return np.asarray(boxes, dtype="int32")

    order = sorted(range(len(boxes)), key=lambda i: -(boxes[i][2] * boxes[i][3]))
    keep = []
    for i in order:
        dominated = False
        for j in keep:
            _, ov_iou, ov_cont = _geometry(boxes[i], boxes[j])
            if ov_iou > iou_threshold or ov_cont > containment_threshold:
                dominated = True
                break
        if not dominated:
            keep.append(i)

    return np.asarray([boxes[i] for i in sorted(keep)], dtype="int32")


class FaceTracker:
    """
    Associates detections across frames, then smooths each face independently.

    Smoothing has to be per-face: one global EMA would average a newborn and a
    grandparent together the moment two people share the frame.

    window        : EMA window in frames; alpha = 2/(window+1). window=1 gives
                    alpha=1, i.e. the raw value -- that is what --no-smooth uses,
                    so tracking/hysteresis stay on while smoothing vanishes.
    match_iou     : minimum IoU for a detection to be considered the same face
                    it was last frame. 0.05 tolerates Haar boxes jumping around
                    a bit while still separating two adjacent people.
    max_age       : frames a face may go undetected before its state is dropped,
                    so a person leaving the frame does not haunt the next one.
    gender_margin : half-width of the hysteresis band around 0.5. A face already
                    labelled Male stays Male until the score clears 0.5 + margin.
                    0.0 reproduces a plain 0.5 threshold exactly.
    """

    def __init__(self, window=5, match_iou=0.05, max_age=10, gender_margin=0.05):
        window = max(1, int(window))
        self.window = window
        self.alpha = 2.0 / (window + 1.0)
        self.match_iou = float(match_iou)
        self.max_age = int(max_age)
        self.gender_margin = max(0.0, float(gender_margin))
        self.smoothing = window > 1
        self._tracks = {}
        self._next_id = 0

    def reset(self):
        self._tracks.clear()
        self._next_id = 0

    @property
    def active_tracks(self):
        return len(self._tracks)

    def _label(self, previous, score):
        lo = 0.5 - self.gender_margin
        hi = 0.5 + self.gender_margin
        if previous == "Male":
            return "Female" if score > hi else "Male"
        if previous == "Female":
            return "Male" if score < lo else "Female"
        if score >= hi:
            return "Female"
        if score <= lo:
            return "Male"
        return "Uncertain"

    def _assign(self, boxes):
        """Greedy best-IoU matching of this frame's boxes to live tracks."""
        candidates = []
        for i, box in enumerate(boxes):
            for tid, tr in self._tracks.items():
                ov = iou(box, tr["box"])
                if ov > self.match_iou:
                    candidates.append((ov, i, tid))
        candidates.sort(reverse=True)

        used_boxes, used_tracks, assign = set(), set(), {}
        for ov, i, tid in candidates:
            if i in used_boxes or tid in used_tracks:
                continue
            used_boxes.add(i)
            used_tracks.add(tid)
            assign[i] = tid
        return assign

    def update(self, faces, gender_scores, ages):
        """
        Feed one frame. Returns a list of dicts, one per (already NMS-merged)
        detection, ordered like the input:

            {"id", "box", "gender", "gender_score", "age",
             "raw_gender_score", "raw_age"}

        gender_score/age are smoothed; the raw_* keys are what --debug prints
        beside them.
        """
        boxes = _to_boxes(faces)
        if not boxes:
            for tr in self._tracks.values():
                tr["misses"] += 1
            self._expire()
            return []

        assign = self._assign(boxes)
        touched = set()
        rows = []

        for i, box in enumerate(boxes):
            tid = assign.get(i)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "box": box, "gender": None, "age": None,
                    "label": None, "misses": 0, "hits": 0,
                }
            tr = self._tracks[tid]

            raw_g = float(gender_scores[i])
            raw_a = float(ages[i])

            if tr["gender"] is None or not self.smoothing:
                tr["gender"], tr["age"] = raw_g, raw_a
            else:
                a = self.alpha
                tr["gender"] = a * raw_g + (1.0 - a) * tr["gender"]
                tr["age"] = a * raw_a + (1.0 - a) * tr["age"]

            tr["box"] = box
            tr["label"] = self._label(tr["label"], tr["gender"])
            tr["hits"] += 1
            tr["misses"] = 0
            touched.add(tid)

            rows.append(
                {
                    "id": tid,
                    "box": box,
                    "gender": tr["label"],
                    "gender_score": tr["gender"],
                    "age": tr["age"],
                    "raw_gender_score": raw_g,
                    "raw_age": raw_a,
                }
            )

        for tid, tr in self._tracks.items():
            if tid not in touched:
                tr["misses"] += 1
        self._expire()
        return rows

    def _expire(self):
        for tid in [t for t, tr in self._tracks.items() if tr["misses"] > self.max_age]:
            del self._tracks[tid]


def format_label(row):
    """
    The label text, formatted identically for the window overlay and the console.

    Truncation, not rounding -- max(0, int(age)) is what this project has always
    done, and keeping it means --no-smooth reproduces the old labels exactly.
    """
    age = max(0, int(row["age"]))
    return f"{row['gender']}, {age} ({row['gender_score']:.2f})"
