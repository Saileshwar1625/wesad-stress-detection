"""
Print the ORDER of protocol segments (baseline / stress / amusement /
meditation) for every subject, from the raw label array.

Why: S14's temperature rises under stress while everyone else's falls
(temp_only balanced accuracy 0.13 for S14). One cheap alternative
explanation is session drift -- skin temperature creeping up over the
recording -- combined with S14 doing stress LATER in the session than
others. If S14's segment order differs from the majority, treat the
temperature reversal as possibly an ordering artifact, not physiology.

Usage:
    python pipeline/segment_order.py --input data/raw
Writes reports/segment_order.txt
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from load_wesad import LABEL_SAMPLING_HZ, load_subject_pickle

NAMES = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
MIN_SEGMENT_SECONDS = 60  # ignore short transients


def segment_sequence(label_array, min_seconds: int = MIN_SEGMENT_SECONDS):
    """[(name, start_sec, duration_sec), ...] for runs of labels 1-4 that last
    at least min_seconds, in time order."""
    label_array = np.asarray(label_array)
    segs = []
    start = 0
    for i in range(1, len(label_array) + 1):
        if i == len(label_array) or label_array[i] != label_array[start]:
            code = int(label_array[start])
            dur = (i - start) / LABEL_SAMPLING_HZ
            if code in NAMES and dur >= min_seconds:
                segs.append((NAMES[code], round(start / LABEL_SAMPLING_HZ, 1), round(dur, 1)))
            start = i
    return segs


def order_string(segs):
    return " > ".join(name for name, _, _ in segs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--report", default="reports/segment_order.txt")
    args = parser.parse_args()

    orders = {}
    for f in sorted(Path(args.input).rglob("S*.pkl"), key=lambda p: p.name):
        data = load_subject_pickle(f)
        orders[f.stem] = segment_sequence(data["label"])

    counts = Counter(order_string(s) for s in orders.values())
    lines = ["=== Segment order per subject (name, start s, duration s) ==="]
    for sid, segs in orders.items():
        lines.append(f"{sid}: {order_string(segs)}   {segs}")
    lines += ["", "=== Order frequency ==="]
    for order, n in counts.most_common():
        subs = [sid for sid, s in orders.items() if order_string(s) == order]
        lines.append(f"{n} subjects: {order}   -> {subs}")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
