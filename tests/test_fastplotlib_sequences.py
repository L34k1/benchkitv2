import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "desktop"))

from _fastplotlib_common import make_sequence


class TestFastplotlibSequences(unittest.TestCase):
    def test_deterministic_sequence(self) -> None:
        initial = {"x_min": 0.0, "x_max": 10.0}
        seq1 = make_sequence("PAN_ZOOM", 10, 0, initial, {})
        seq2 = make_sequence("PAN_ZOOM", 10, 0, initial, {})
        self.assertEqual(seq1, seq2)

    def test_op_counts(self) -> None:
        initial = {"x_min": 0.0, "x_max": 10.0}
        seq = make_sequence("PAN_ZOOM", 5, 0, initial, {})
        ops = [s.op for s in seq]
        self.assertEqual(ops, ["PAN", "ZOOM_IN", "PAN", "ZOOM_IN", "PAN"])


if __name__ == "__main__":
    unittest.main()
