from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.db import connect, utc_now
from app.services.machine_review import run_machine_review_job


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine-review-job-id", type=int, required=True)
    args = parser.parse_args()
    try:
        result = run_machine_review_job(args.machine_review_job_id)
    except Exception as exc:
        now = utc_now()
        with connect() as conn:
            conn.execute(
                """
                UPDATE machine_review_jobs
                SET status = 'failed', ended_at = ?, updated_at = ?, return_code = 1,
                    error_message = COALESCE(NULLIF(error_message, ''), ?)
                WHERE id = ?
                """,
                (now, now, str(exc), args.machine_review_job_id),
            )
        print(f"Machine Review Job #{args.machine_review_job_id} failed: {exc}", flush=True)
        return 1
    print(
        "Machine Review Job #{job_id} finished: status={status}, processed={processed}, scored={scored}, failed={failed}".format(
            **result
        ),
        flush=True,
    )
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
