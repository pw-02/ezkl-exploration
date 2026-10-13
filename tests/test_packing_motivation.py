import unittest

from analysis.section_3.packing_motivation import oracle_rows, scaling_rows
from analysis.section_3.simulate_scheduling import MeasuredSegment, TraceJob


def job(job_id: str, segments):
    return TraceJob(
        request_id="request",
        job_id=job_id,
        workload="model",
        segments=tuple(
            MeasuredSegment(stage, stage, duration, memory)
            for stage, duration, memory in segments
        ),
    )


class PackingMotivationTests(unittest.TestCase):
    def test_stage_accounting_exposes_overlap_hidden_by_job_peaks(self):
        jobs = [
            job("a", [("LOW", 10, 2), ("HIGH", 10, 8)]),
            job("b", [("HIGH", 10, 8), ("LOW", 10, 2)]),
        ]
        rows = scaling_rows(jobs, memory_budget_gb=10, replicas=[1])
        by_model = {row["memory_model"]: row for row in rows}
        self.assertGreater(
            by_model["job_peak"]["makespan_s"],
            by_model["stage_peak"]["makespan_s"],
        )
        self.assertGreater(by_model["stage_peak"]["makespan_saved_percent"], 0)
        self.assertGreater(by_model["stage_peak"]["speedup_vs_job_peak"], 1)

    def test_small_oracle_labels_only_completed_search_as_optimal(self):
        jobs = [
            job("a", [("LOW", 1, 2), ("HIGH", 1, 8)]),
            job("b", [("HIGH", 1, 8), ("LOW", 1, 2)]),
        ]
        rows = oracle_rows(
            jobs,
            memory_budget_gb=10,
            source_job_limit=2,
            time_limit_s=2,
            max_phases=8,
        )
        self.assertTrue(rows)
        self.assertTrue(all(row["optimality_proven"] for row in rows))
        self.assertTrue(all(row["status"] == "complete" for row in rows))


if __name__ == "__main__":
    unittest.main()
