"""Run directly: python pipeline/test_segment_order.py"""
import numpy as np
from load_wesad import LABEL_SAMPLING_HZ
from segment_order import segment_sequence, order_string


def test_sequence_and_transient_filtering():
    hz = LABEL_SAMPLING_HZ
    label = np.concatenate([
        np.full(120 * hz, 1), np.full(10 * hz, 0),   # baseline, short transient (ignored)
        np.full(90 * hz, 2), np.full(100 * hz, 3),   # stress, amusement
        np.full(5 * hz, 4),                          # too-short meditation (ignored)
    ])
    segs = segment_sequence(label)
    assert order_string(segs) == "baseline > stress > amusement", segs
    assert segs[1][1] == 130.0 and segs[1][2] == 90.0, segs
    print("test_sequence_and_transient_filtering: OK")


if __name__ == "__main__":
    test_sequence_and_transient_filtering()
    print("\nAll segment_order.py sanity tests passed.")
