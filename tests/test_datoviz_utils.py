import unittest

from benchkit.datoviz_bench_utils import build_step_rows, build_step_specs, summarize_latency_s


class TestDatovizBenchUtils(unittest.TestCase):
    def test_step_rows_count_matches_steps(self) -> None:
        specs = build_step_specs("PAN", lo=0.0, hi=10.0, window_s=1.0, steps=5, y0=-1.0, y1=2.0)
        issue_times = [0.0] * 5
        ack_times = [0.1] * 5
        statuses = ["OK"] * 5
        rows = build_step_rows(specs, issue_times, ack_times, statuses)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["step_index"], 0)
        self.assertEqual(rows[-1]["step_index"], 4)

    def test_summarize_latency(self) -> None:
        summary = summarize_latency_s([0.1, 0.2, 0.3])
        self.assertAlmostEqual(summary["latency_p50_s"], 0.2)
        self.assertAlmostEqual(summary["latency_p95_s"], 0.29)
        self.assertAlmostEqual(summary["latency_max_s"], 0.3)
        self.assertEqual(summary["count"], 3)


if __name__ == "__main__":
    unittest.main()
