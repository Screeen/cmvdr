import unittest
import csv
import tempfile
from pathlib import Path

from script.benchmark_runtime import _aggregate_rows, compare_summaries


class BenchmarkRuntimeTests(unittest.TestCase):
    def test_aggregate_rows_computes_means_and_slowdown(self):
        rows = [
            {"dataset": "Synthetic", "algorithm": "mpdr", "runtime_s": 2.0},
            {"dataset": "Synthetic", "algorithm": "mpdr", "runtime_s": 3.0},
            {"dataset": "MUSAN", "algorithm": "mpdr", "runtime_s": 1.0},
        ]

        summary = _aggregate_rows(rows)
        synthetic = next(row for row in summary if row["dataset"] == "Synthetic")
        overall = next(row for row in summary if row["dataset"] == "Overall")

        self.assertAlmostEqual(synthetic["runtime_mean_s"], 2.5)
        self.assertEqual(synthetic["samples"], 2)
        self.assertEqual(overall["samples"], 3)

    def test_compare_summaries_combines_runtime_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            mpdr_path = tmpdir / "mpdr.csv"
            cmpdr_path = tmpdir / "cmpdr.csv"
            out_path = tmpdir / "table.csv"

            with mpdr_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["dataset", "algorithm", "runtime_mean_s", "samples"])
                writer.writeheader()
                writer.writerow({"dataset": "Synthetic", "algorithm": "mpdr", "runtime_mean_s": 2.0, "samples": 2})

            with cmpdr_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["dataset", "algorithm", "runtime_mean_s", "samples"])
                writer.writeheader()
                writer.writerow({"dataset": "Synthetic", "algorithm": "cmpdr", "runtime_mean_s": 4.0, "samples": 2})

            compare_summaries(mpdr_path, cmpdr_path, out_path)
            self.assertTrue(out_path.exists())
            self.assertIn("Overall", out_path.read_text())


if __name__ == "__main__":
    unittest.main()
