import re
from datetime import datetime, timezone

from app.services.operation_monitor import completion_eta_label, format_display_datetime, operation_monitor


def test_operation_monitor_treats_candidate_comparison_stages_as_running():
    monitor = operation_monitor(
        operation_type="candidate_standard_comparison",
        type_label="Standard Candidate Comparison",
        status="generating_images",
        started_at="2026-06-26T00:00:00+00:00",
        progress_current=0,
        progress_total=117,
        status_url="/jobs/106/candidate-comparisons/5/status",
    )

    assert monitor["is_running"] is True


def test_operation_monitor_treats_completed_as_not_running():
    monitor = operation_monitor(
        operation_type="candidate_standard_comparison",
        type_label="Standard Candidate Comparison",
        status="completed",
        started_at="2026-06-26T00:00:00+00:00",
        progress_current=117,
        progress_total=117,
        status_url="/jobs/106/candidate-comparisons/5/status",
    )

    assert monitor["is_running"] is False


def test_operation_monitor_formats_visible_datetimes_consistently():
    monitor = operation_monitor(
        operation_type="training",
        type_label="Training",
        status="running",
        started_at="2026-06-26T10:16:49+00:00",
    )

    assert re.match(r"2026-06-26 \d\d:\d\d:\d\d UTC[+-]\d\d:\d\d", monitor["started_at"])


def test_completion_eta_uses_display_datetime_format():
    label = completion_eta_label(0, datetime(2026, 6, 26, 10, 16, 49, tzinfo=timezone.utc))

    assert re.match(r"2026-06-26 \d\d:\d\d:\d\d UTC[+-]\d\d:\d\d", label)


def test_format_display_datetime_handles_missing_values():
    assert format_display_datetime(None) == "-"
