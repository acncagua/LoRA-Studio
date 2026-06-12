from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import connect, init_db, utc_now


def main() -> None:
    init_db()
    now = utc_now()
    with connect() as conn:
        preset = conn.execute("SELECT id FROM validation_presets WHERE id = 'standard_validation_v1'").fetchone()
        profile = conn.execute("SELECT id FROM selected_lora_profiles WHERE id = 1 AND job_id = 12").fetchone()
        if not preset or not profile:
            print("Job #12 profile or standard_validation_v1 was not found. No changes made.")
            return
        conn.execute(
            """
            UPDATE selected_lora_profiles
            SET default_validation_preset_id = COALESCE(default_validation_preset_id, 'standard_validation_v1'),
                validation_policy_memo = COALESCE(
                    NULLIF(validation_policy_memo, ''),
                    '通常比較はHiresなしのStandard Validationを基準にする。HiresありはExtended Validationで最終見栄え確認として扱う。'
                ),
                updated_at = ?
            WHERE id = 1 AND job_id = 12
            """,
            (now,),
        )
    print("Applied one-off Job #12 validation defaults.")


if __name__ == "__main__":
    main()
