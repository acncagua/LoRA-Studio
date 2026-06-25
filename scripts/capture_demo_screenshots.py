from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def demo_ids(db_path: Path) -> dict[str, int]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        project = conn.execute("SELECT id FROM lora_projects WHERE name = 'Demo Character LoRA'").fetchone()
        job = conn.execute("SELECT id FROM training_jobs WHERE name = 'Demo Training Job completed'").fetchone()
        session = conn.execute("SELECT id FROM review_sessions WHERE name = 'Candidate Epoch Review completed'").fetchone()
        validation = conn.execute("SELECT id FROM validation_runs WHERE validation_run_kind = 'weight_calibration' ORDER BY id DESC LIMIT 1").fetchone()
    if not all([project, job, session, validation]):
        raise SystemExit("Demo DB does not contain the expected seeded records. Run scripts/create_demo_db.py first.")
    return {
        "project": int(project["id"]),
        "job": int(job["id"]),
        "session": int(session["id"]),
        "validation": int(validation["id"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture sanitized English demo screenshots from a running LoRA-Studio server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8768")
    parser.add_argument("--db", default="demo/demo.sqlite")
    parser.add_argument("--output", default="demo/screenshots/generated")
    args = parser.parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit("Playwright is not installed. Install it or capture screenshots manually from the demo server.") from exc

    db_path = (ROOT_DIR / args.db).resolve() if not Path(args.db).is_absolute() else Path(args.db).resolve()
    ids = demo_ids(db_path)
    out_dir = (ROOT_DIR / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "dashboard": "/?lang=en",
        "project": f"/projects/{ids['project']}?lang=en",
        "job": f"/jobs/{ids['job']}?lang=en",
        "job_wizard_purpose": "/jobs/new?mode=purpose&lang=en",
        "review_session": f"/review-sessions/{ids['session']}?lang=en",
        "validation_run": f"/validation-runs/{ids['validation']}?lang=en",
        "lora_library": "/lora-library?lang=en",
        "training_recipes": "/training-recipes?lang=en",
        "optimizers": "/optimizers?lang=en",
    }

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1200})
        for name, path in targets.items():
            page.goto(args.base_url.rstrip("/") + path, wait_until="networkidle")
            page.screenshot(path=str(out_dir / f"{name}.png"), full_page=True)
        browser.close()
    print(f"Captured {len(targets)} screenshots under {out_dir}")


if __name__ == "__main__":
    main()
