from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app import settings
from app.db import connect, fetch_all, fetch_one, utc_now
from app.services.lora_artifacts import ResolvedLoraArtifact, resolve_lora_artifact
from app.services.validation_generation import build_run_cross_matrix_html, start_missing_validation_review_sequences
from app.services.validation_runs import add_validation_run_weights, create_validation_run


COMPARISON_MODES = {"controlled", "practical"}
COMPARISON_AXES = {"network_type", "optimizer_profile", "recipe", "selected_artifact"}
DECISION_STATUSES = {"human_review_pending", "candidate_preferred", "no_clear_winner", "retest_required"}


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _profile_row(profile_id: int) -> dict[str, Any]:
    row = fetch_one(
        """
        SELECT p.*, j.name AS job_name, j.dataset_id, j.dataset_version_id,
               j.base_model_path, j.model_family, j.recipe_v2_id,
               j.optimizer_definition_id, j.optimizer_profile_id,
               j.network_type_id, j.training_purpose_id, j.params_snapshot_json,
               j.recipe_snapshot_json, j.user_overrides_json,
               o.epoch AS output_epoch, o.step AS output_step, o.sha256 AS output_sha256,
               o.file_size AS output_file_size, o.file_path AS output_file_path,
               nt.display_name AS network_type_name,
               op.display_name AS optimizer_profile_name,
               tr.display_name AS recipe_name
        FROM selected_lora_profiles p
        LEFT JOIN training_jobs j ON j.id = p.job_id
        LEFT JOIN training_outputs o ON o.id = p.selected_output_id
        LEFT JOIN network_type_definitions nt ON nt.id = j.network_type_id
        LEFT JOIN optimizer_profiles_v2 op ON op.id = j.optimizer_profile_id
        LEFT JOIN training_recipes_v2 tr ON tr.id = j.recipe_v2_id
        WHERE p.id = ?
        """,
        (profile_id,),
    )
    if row is None:
        raise ValueError(f"Selected LoRA profile not found: {profile_id}")
    return dict(row)


def _display_label(profile: dict[str, Any]) -> str:
    if profile.get("profile_name"):
        return str(profile["profile_name"])
    network = profile.get("network_type_name") or profile.get("network_type_id") or "LoRA"
    optimizer = profile.get("optimizer_profile_name") or profile.get("optimizer_profile_id") or ""
    label = f"{network} {optimizer}".strip()
    if not label:
        label = f"Job #{profile['job_id']}"
    return label


def _display_detail(profile: dict[str, Any], artifact: ResolvedLoraArtifact | None = None) -> str:
    parts = [
        f"Profile #{int(profile['id'])}",
        f"Job #{int(profile['job_id'])}",
    ]
    if profile.get("network_type_name") or profile.get("network_type_id"):
        parts.append(f"Network: {profile.get('network_type_name') or profile.get('network_type_id')}")
    if profile.get("optimizer_profile_name") or profile.get("optimizer_profile_id"):
        parts.append(f"Optimizer: {profile.get('optimizer_profile_name') or profile.get('optimizer_profile_id')}")
    if profile.get("recipe_name") or profile.get("recipe_v2_id"):
        parts.append(f"Recipe: {profile.get('recipe_name') or profile.get('recipe_v2_id')}")
    if profile.get("output_epoch") is not None:
        parts.append(f"epoch {profile['output_epoch']}")
    if profile.get("output_step") is not None:
        parts.append(f"step {profile['output_step']}")
    if artifact is not None:
        parts.append(f"artifact: {artifact.source_kind}")
    return " / ".join(parts)


def build_candidate_snapshot(profile: dict[str, Any], artifact: ResolvedLoraArtifact) -> dict[str, Any]:
    params = _json_load(profile.get("params_snapshot_json"), {})
    return {
        "profile": {
            "id": profile.get("id"),
            "profile_name": profile.get("profile_name"),
            "project_id": profile.get("project_id"),
            "trigger_word": profile.get("trigger_word"),
            "base_model": profile.get("base_model"),
            "recommended_weight_min": profile.get("recommended_weight_min"),
            "recommended_weight_max": profile.get("recommended_weight_max"),
        },
        "job": {
            "id": profile.get("job_id"),
            "name": profile.get("job_name"),
            "model_family": profile.get("model_family"),
            "dataset_id": profile.get("dataset_id"),
            "dataset_version_id": profile.get("dataset_version_id"),
            "base_model_path": profile.get("base_model_path"),
            "recipe_v2_id": profile.get("recipe_v2_id"),
            "optimizer_definition_id": profile.get("optimizer_definition_id"),
            "optimizer_profile_id": profile.get("optimizer_profile_id"),
            "network_type_id": profile.get("network_type_id"),
            "training_purpose_id": profile.get("training_purpose_id"),
            "recipe_snapshot_json": _json_load(profile.get("recipe_snapshot_json"), {}),
            "params_snapshot_json": params,
            "user_overrides_json": _json_load(profile.get("user_overrides_json"), {}),
            "training_seed": params.get("seed"),
            "training_seed_status": "recorded" if params.get("seed") is not None else "not_recorded",
        },
        "output": {
            "id": profile.get("selected_output_id"),
            "epoch": profile.get("output_epoch"),
            "step": profile.get("output_step"),
            "file_size": profile.get("output_file_size"),
            "sha256": profile.get("output_sha256"),
        },
        "artifact": {
            "resolved_path": str(artifact.path),
            "source_kind": artifact.source_kind,
            "sha256": artifact.actual_sha256,
            "file_size": artifact.file_size,
            "warnings": list(artifact.warnings),
        },
    }


def _candidate_value(candidate: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = candidate.get("snapshot") or {}
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def run_parity_gate(candidates: list[dict[str, Any]], mode: str, axis: str) -> dict[str, Any]:
    report = {
        "status": "pass",
        "mode": mode,
        "axis": axis,
        "shared_values": {},
        "allowed_differences": [],
        "unexpected_differences": [],
        "warnings": [],
        "errors": [],
    }
    if len(candidates) < 2 or len(candidates) > 6:
        report["errors"].append("Candidate count must be between 2 and 6.")
    project_ids = {candidate["profile"].get("project_id") for candidate in candidates}
    if len(project_ids) != 1 or None in project_ids:
        report["errors"].append("All selected LoRA profiles must belong to the same Project.")
    model_families = {_candidate_value(candidate, ("job", "model_family")) for candidate in candidates}
    if len(model_families) > 1:
        report["errors"].append("Model family differs between candidates.")
    base_models = {_candidate_value(candidate, ("job", "base_model_path")) or _candidate_value(candidate, ("profile", "base_model")) for candidate in candidates}
    if len(base_models) > 1:
        report["errors"].append("Base model differs between candidates.")
    if not all(candidate.get("artifact") and candidate["artifact"].verified for candidate in candidates):
        report["errors"].append("All candidates must have a verified LoRA artifact.")

    compare_paths = [
        ("job", "dataset_id"),
        ("job", "dataset_version_id"),
        ("job", "optimizer_definition_id"),
        ("job", "optimizer_profile_id"),
        ("job", "network_type_id"),
        ("job", "recipe_v2_id"),
        ("job", "training_seed"),
    ]
    allowed_by_axis = {
        "network_type": {("job", "network_type_id"), ("job", "recipe_v2_id")},
        "optimizer_profile": {("job", "optimizer_definition_id"), ("job", "optimizer_profile_id")},
        "recipe": {("job", "recipe_v2_id")},
        "selected_artifact": set(),
    }
    if mode == "practical":
        allowed = set(compare_paths)
    else:
        allowed = allowed_by_axis.get(axis, set())
    for path in compare_paths:
        values = [_candidate_value(candidate, path) for candidate in candidates]
        key = ".".join(path)
        distinct = {str(value) for value in values if value is not None}
        if len(distinct) <= 1:
            report["shared_values"][key] = values[0] if values else None
            continue
        entry = {"field": key, "values": values}
        if path in allowed:
            report["allowed_differences"].append(entry)
        else:
            report["unexpected_differences"].append(entry)
    seeds = [_candidate_value(candidate, ("job", "training_seed")) for candidate in candidates]
    if any(seed is None for seed in seeds):
        report["warnings"].append("Training seed is not recorded for at least one candidate.")
    if report["errors"] or report["unexpected_differences"]:
        report["status"] = "fail"
    elif report["warnings"] or report["allowed_differences"]:
        report["status"] = "warning"
    return report


def _fingerprint_row(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("prompt_key") or "prompt",
        row.get("prompt") or "",
        row.get("negative_prompt") or "",
        row.get("trigger_word") or "",
        int(row.get("seed") or 0),
        f"{float(row.get('lora_weight') or 0):g}",
        int(row.get("hires_enabled") or 0),
        int(row.get("width") or 0),
        int(row.get("height") or 0),
        row.get("sampler") or "",
        int(row.get("steps") or 0),
        float(row.get("cfg_scale") or 0),
        row.get("base_model") or "",
    )


def condition_fingerprint_set(run_id: int) -> set[tuple[Any, ...]]:
    return {
        _fingerprint_row(dict(row))
        for row in fetch_all(
            "SELECT * FROM validation_expected_conditions WHERE validation_run_id = ?",
            (run_id,),
        )
    }


def condition_sets_match(run_ids: list[int]) -> tuple[bool, dict[int, int]]:
    counts: dict[int, int] = {}
    baseline: set[tuple[Any, ...]] | None = None
    for run_id in run_ids:
        fingerprints = condition_fingerprint_set(run_id)
        counts[int(run_id)] = len(fingerprints)
        if baseline is None:
            baseline = fingerprints
        elif fingerprints != baseline:
            return False, counts
    return True, counts


def _backfill_validation_run_artifact(run_id: int, artifact: ResolvedLoraArtifact) -> None:
    with connect() as conn:
        conn.execute(
            """
            UPDATE validation_runs
            SET artifact_path_snapshot = COALESCE(artifact_path_snapshot, ?),
                artifact_source_kind = COALESCE(artifact_source_kind, ?),
                artifact_sha256_snapshot = COALESCE(artifact_sha256_snapshot, ?),
                artifact_file_size_snapshot = COALESCE(artifact_file_size_snapshot, ?),
                updated_at = ?
            WHERE id = ?
            """,
            (str(artifact.path), artifact.source_kind, artifact.actual_sha256, artifact.file_size, utc_now(), run_id),
        )


def compatible_validation_run(profile: dict[str, Any], validation_preset_id: str, artifact: ResolvedLoraArtifact) -> dict[str, Any] | None:
    rows = fetch_all(
        """
        SELECT *
        FROM validation_runs
        WHERE validation_preset_id = ?
          AND (
            selected_lora_profile_id = ?
            OR selected_output_id = ?
            OR (job_id = ? AND selected_output_id = ?)
          )
        ORDER BY updated_at DESC, id DESC
        """,
        (
            validation_preset_id,
            profile.get("id"),
            profile.get("selected_output_id"),
            profile.get("job_id"),
            profile.get("selected_output_id"),
        ),
    )
    for row in rows:
        run = dict(row)
        snapshot_sha = run.get("artifact_sha256_snapshot")
        if snapshot_sha and snapshot_sha != artifact.actual_sha256:
            continue
        if not snapshot_sha:
            try:
                current = resolve_lora_artifact(
                    profile_id=run.get("selected_lora_profile_id") or profile.get("id"),
                    output_id=run.get("selected_output_id") or profile.get("selected_output_id"),
                    job_id=run.get("job_id") or profile.get("job_id"),
                )
            except ValueError:
                continue
            if current.actual_sha256 != artifact.actual_sha256:
                continue
            _backfill_validation_run_artifact(int(run["id"]), current)
            run["artifact_sha256_snapshot"] = current.actual_sha256
        return run
    return None


def _preset_snapshot(validation_preset_id: str) -> str:
    preset = fetch_one("SELECT * FROM validation_presets WHERE id = ?", (validation_preset_id,))
    if preset is None:
        raise ValueError(f"Validation preset not found: {validation_preset_id}")
    return _json_dump(dict(preset))


def create_lora_comparison_session(
    *,
    profile_ids: list[int],
    name: str,
    comparison_mode: str,
    comparison_axis: str,
    validation_preset_id: str,
    allow_warnings: bool = False,
    memo: str = "",
) -> int:
    profile_ids = list(dict.fromkeys(int(profile_id) for profile_id in profile_ids if int(profile_id) > 0))
    if comparison_mode not in COMPARISON_MODES:
        raise ValueError("Unknown comparison mode.")
    if comparison_axis not in COMPARISON_AXES:
        raise ValueError("Unknown comparison axis.")
    if len(profile_ids) < 2:
        raise ValueError("LoRA比較には2件以上のProfileを選択してください。")
    if len(profile_ids) > 6:
        raise ValueError("LoRA比較は最大6件までです。")

    candidates: list[dict[str, Any]] = []
    for profile_id in profile_ids:
        profile = _profile_row(profile_id)
        artifact = resolve_lora_artifact(profile_id=profile_id)
        snapshot = build_candidate_snapshot(profile, artifact)
        candidates.append({"profile": profile, "artifact": artifact, "snapshot": snapshot})
    parity = run_parity_gate(candidates, comparison_mode, comparison_axis)
    if parity["status"] == "fail":
        raise ValueError("Parity Gate failed: " + "; ".join(parity["errors"] + [d["field"] for d in parity["unexpected_differences"]]))
    if parity["status"] == "warning" and not allow_warnings:
        raise ValueError("Parity Gate warning. 警告を確認してから作成してください。")

    preset_snapshot = _preset_snapshot(validation_preset_id)
    project_id = int(candidates[0]["profile"]["project_id"])
    now = utc_now()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO lora_comparison_sessions(
                project_id, name, comparison_mode, comparison_axis,
                validation_preset_id, preset_snapshot_json,
                status, parity_status, parity_report_json,
                candidate_count, created_at, updated_at, memo
            )
            VALUES (?, ?, ?, ?, ?, ?, 'ready', ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                name.strip() or "Selected LoRA Comparison",
                comparison_mode,
                comparison_axis,
                validation_preset_id,
                preset_snapshot,
                parity["status"],
                _json_dump(parity),
                len(candidates),
                now,
                now,
                memo.strip(),
            ),
        )
        session_id = int(cur.lastrowid)
    try:
        for index, candidate in enumerate(candidates, start=1):
            profile = candidate["profile"]
            artifact: ResolvedLoraArtifact = candidate["artifact"]
            run = compatible_validation_run(profile, validation_preset_id, artifact)
            run_source = "reused"
            if run is None:
                run_id = create_validation_run(
                    int(profile["job_id"]),
                    validation_preset_id=validation_preset_id,
                    base_model=str(profile.get("base_model") or profile.get("base_model_path") or ""),
                    trigger_word=str(profile.get("trigger_word") or ""),
                    memo=f"LoRA comparison session #{session_id}",
                    profile_id=int(profile["id"]),
                    selected_output_id=int(profile["selected_output_id"]) if profile.get("selected_output_id") else None,
                )
                with connect() as conn:
                    conn.execute(
                        "UPDATE validation_runs SET validation_run_kind = 'lora_comparison', updated_at = ? WHERE id = ?",
                        (utc_now(), run_id),
                    )
                run_source = "created"
            else:
                run_id = int(run["id"])
            with connect() as conn:
                conn.execute(
                    """
                    INSERT INTO lora_comparison_candidates(
                        comparison_session_id, sort_order, selected_lora_profile_id,
                        job_id, selected_output_id, validation_run_id, validation_run_source,
                        display_label_snapshot, display_detail_snapshot,
                        artifact_path_snapshot, artifact_source_kind, artifact_sha256,
                        artifact_file_size, artifact_verified_at, candidate_snapshot_json,
                        allowed_diff_json, unexpected_diff_json, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        index,
                        int(profile["id"]),
                        int(profile["job_id"]),
                        int(profile["selected_output_id"]) if profile.get("selected_output_id") else None,
                        run_id,
                        run_source,
                        _display_label(profile),
                        _display_detail(profile, artifact),
                        str(artifact.path),
                        artifact.source_kind,
                        artifact.actual_sha256,
                        artifact.file_size,
                        now,
                        _json_dump(candidate["snapshot"]),
                        _json_dump(parity.get("allowed_differences", [])),
                        _json_dump(parity.get("unexpected_differences", [])),
                        now,
                        now,
                    ),
                )
    except Exception:
        with connect() as conn:
            conn.execute("DELETE FROM lora_comparison_candidates WHERE comparison_session_id = ?", (session_id,))
            conn.execute("DELETE FROM lora_comparison_sessions WHERE id = ?", (session_id,))
        raise
    refresh_lora_comparison_session(session_id)
    return session_id


def lora_comparison_sessions() -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in fetch_all(
            """
            SELECT s.*, p.name AS project_name
            FROM lora_comparison_sessions s
            LEFT JOIN lora_projects p ON p.id = s.project_id
            ORDER BY s.updated_at DESC, s.id DESC
            """
        )
    ]


def load_lora_comparison_session(session_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    session = fetch_one(
        """
        SELECT s.*, p.name AS project_name
        FROM lora_comparison_sessions s
        LEFT JOIN lora_projects p ON p.id = s.project_id
        WHERE s.id = ?
        """,
        (session_id,),
    )
    if session is None:
        raise ValueError(f"LoRA comparison session not found: {session_id}")
    candidates = [
        dict(row)
        for row in fetch_all(
            """
            SELECT c.*, vr.status AS validation_status, vr.actual_image_count,
                   vr.expected_image_count, vr.name AS validation_run_name
            FROM lora_comparison_candidates c
            LEFT JOIN validation_runs vr ON vr.id = c.validation_run_id
            WHERE c.comparison_session_id = ?
            ORDER BY c.sort_order, c.id
            """,
            (session_id,),
        )
    ]
    return dict(session), candidates


def refresh_lora_comparison_session(session_id: int) -> dict[str, Any]:
    session, candidates = load_lora_comparison_session(session_id)
    run_ids = [int(candidate["validation_run_id"]) for candidate in candidates if candidate.get("validation_run_id")]
    logical = 0
    registered = 0
    reviewed = 0
    for run_id in run_ids:
        row = fetch_one("SELECT COUNT(*) AS count FROM validation_expected_conditions WHERE validation_run_id = ?", (run_id,))
        logical += int(row["count"] or 0) if row else 0
        row = fetch_one("SELECT COUNT(*) AS count FROM validation_images WHERE validation_run_id = ? AND image_role = 'individual'", (run_id,))
        registered += int(row["count"] or 0) if row else 0
    if run_ids:
        placeholders = ",".join("?" for _ in run_ids)
        row = fetch_one(
            f"""
            SELECT COUNT(*) AS count
            FROM machine_review_scores
            WHERE source_type = 'validation_image'
              AND validation_run_id IN ({placeholders})
            """,
            tuple(run_ids),
        )
        reviewed = int(row["count"] or 0) if row else 0
    remaining = max(logical - registered, 0)
    status = session["status"]
    if status not in {"completed", "archived"}:
        status = "ready_for_review" if remaining == 0 and logical > 0 else "ready"
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE lora_comparison_sessions
            SET candidate_count = ?, logical_image_count = ?, physical_image_count = ?,
                reused_image_count = ?, remaining_generation_count = ?,
                registered_image_count = ?, machine_review_score_count = ?,
                status = ?, updated_at = ?
            WHERE id = ?
            """,
            (len(candidates), logical, logical, registered, remaining, registered, reviewed, status, now, session_id),
        )
    return {
        "status": status,
        "candidate_count": len(candidates),
        "logical_image_count": logical,
        "registered_image_count": registered,
        "machine_review_score_count": reviewed,
        "remaining_generation_count": remaining,
    }


def _run_label_details(candidates: list[dict[str, Any]]) -> tuple[dict[int, str], dict[int, str]]:
    labels: dict[int, str] = {}
    details: dict[int, str] = {}
    for candidate in candidates:
        if not candidate.get("validation_run_id"):
            continue
        run_id = int(candidate["validation_run_id"])
        labels[run_id] = candidate["display_label_snapshot"]
        details[run_id] = candidate.get("display_detail_snapshot") or ""
    return labels, details


def _matrix_controls(session_id: int, display_weights: list[Any] | None = None) -> str:
    selected = {round(float(weight), 1) for weight in display_weights or [] if str(weight).strip()}
    options = []
    for value in [round(i / 10, 1) for i in range(0, 21)]:
        checked = " checked" if (not selected and value in {0.0, 0.6, 0.8, 1.0}) or value in selected else ""
        options.append(
            f"<label class=\"weight-chip\"><input type=\"checkbox\" name=\"weights\" value=\"{value:g}\"{checked}> {value:g}</label>"
        )
    return (
        "<section class=\"weight-controls\"><h2>weight選択</h2>"
        f"<form method=\"post\" action=\"/lora-comparisons/{session_id}/matrix/weights\">"
        + "".join(options)
        + "<button class=\"button\" type=\"submit\">選択weightを全候補へ追加</button></form>"
        f"<form method=\"post\" action=\"/lora-comparisons/{session_id}/matrix/missing-review\">"
        "<button class=\"button\" type=\"submit\">全候補の不足レビューを再計算</button></form></section>"
    )


def build_lora_comparison_matrix_html(session_id: int, display_weights: list[Any] | None = None) -> str:
    session, candidates = load_lora_comparison_session(session_id)
    run_ids = [int(candidate["validation_run_id"]) for candidate in candidates if candidate.get("validation_run_id")]
    ok, counts = condition_sets_match(run_ids)
    if not ok:
        raise ValueError(f"比較条件が候補間で一致していません: {counts}")
    labels, details = _run_label_details(candidates)
    return build_run_cross_matrix_html(
        run_ids,
        title=f"LoRA Comparison Matrix #{session_id}: {session['name']}",
        run_labels=labels,
        run_details=details,
        display_weights=display_weights,
        navigation_html=(
            "<div class=\"matrix-actions\">"
            f"<a class=\"button\" href=\"/lora-comparisons/{session_id}\">比較詳細へ戻る</a>"
            "<button class=\"button secondary\" type=\"button\" onclick=\"window.close()\">閉じる</button>"
            "</div>"
        ),
        controls_html=_matrix_controls(session_id, display_weights),
    )


def write_lora_comparison_matrix(session_id: int, display_weights: list[Any] | None = None) -> str:
    html = build_lora_comparison_matrix_html(session_id, display_weights)
    out_dir = settings.EXPORTS_DIR / "lora_comparisons" / f"comparison_{session_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "comparison_matrix.html"
    path.write_text(html, encoding="utf-8")
    with connect() as conn:
        conn.execute(
            "UPDATE lora_comparison_sessions SET matrix_path = ?, updated_at = ? WHERE id = ?",
            (str(path), utc_now(), session_id),
        )
    return str(path)


def add_lora_comparison_weights(session_id: int, weights: list[Any]) -> dict[str, Any]:
    _, candidates = load_lora_comparison_session(session_id)
    results = {}
    for candidate in candidates:
        if candidate.get("validation_run_id"):
            results[int(candidate["validation_run_id"])] = add_validation_run_weights(int(candidate["validation_run_id"]), weights)
    refresh_lora_comparison_session(session_id)
    return {"session_id": session_id, "runs": results}


def start_lora_comparison_missing_review(session_id: int) -> dict[str, Any]:
    _, candidates = load_lora_comparison_session(session_id)
    run_ids = [int(candidate["validation_run_id"]) for candidate in candidates if candidate.get("validation_run_id")]
    result = start_missing_validation_review_sequences(run_ids)
    refresh_lora_comparison_session(session_id)
    return result


def save_lora_comparison_decision(
    session_id: int,
    *,
    decision_status: str,
    preferred_candidate_id: int | None,
    decision_reason: str,
) -> None:
    if decision_status not in DECISION_STATUSES:
        raise ValueError("Unknown decision status.")
    session, candidates = load_lora_comparison_session(session_id)
    candidate_ids = {int(candidate["id"]) for candidate in candidates}
    if decision_status == "candidate_preferred":
        if not preferred_candidate_id or int(preferred_candidate_id) not in candidate_ids:
            raise ValueError("Preferred candidate is required.")
        summary = refresh_lora_comparison_session(session_id)
        if summary["remaining_generation_count"] > 0:
            raise ValueError("画像が不足しているため、候補採用判定は保存できません。")
    else:
        preferred_candidate_id = None
    with connect() as conn:
        conn.execute(
            """
            UPDATE lora_comparison_sessions
            SET decision_status = ?, preferred_candidate_id = ?, decision_reason = ?,
                status = CASE WHEN ? = 'human_review_pending' THEN status ELSE 'completed' END,
                updated_at = ?
            WHERE id = ?
            """,
            (decision_status, preferred_candidate_id, decision_reason.strip(), decision_status, utc_now(), session_id),
        )


def write_lora_comparison_report(session_id: int) -> tuple[str, str]:
    session, candidates = load_lora_comparison_session(session_id)
    summary = refresh_lora_comparison_session(session_id)
    out_dir = settings.EXPORTS_DIR / "lora_comparisons" / f"comparison_{session_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "comparison_report.md"
    json_path = out_dir / "comparison_report.json"
    report = {
        "session": session,
        "candidates": candidates,
        "summary": summary,
        "generated_at": utc_now(),
    }
    lines = [
        f"# LoRA Comparison #{session_id}: {session['name']}",
        "",
        f"- Project: {session.get('project_name') or session.get('project_id')}",
        f"- Mode: {session['comparison_mode']}",
        f"- Axis: {session['comparison_axis']}",
        f"- Parity: {session['parity_status']}",
        f"- Decision: {session['decision_status']}",
        f"- Logical images: {summary['logical_image_count']}",
        f"- Registered images: {summary['registered_image_count']}",
        f"- Machine reviewed: {summary['machine_review_score_count']}",
        "",
        "## Candidates",
        "",
    ]
    for candidate in candidates:
        lines.extend(
            [
                f"### {candidate['display_label_snapshot']}",
                "",
                f"- Profile: #{candidate['selected_lora_profile_id']}",
                f"- Job: #{candidate['job_id']}",
                f"- Validation Run: #{candidate['validation_run_id']}",
                f"- Artifact: `{candidate['artifact_path_snapshot']}`",
                f"- SHA-256: `{candidate['artifact_sha256']}`",
                f"- Size: {candidate['artifact_file_size']}",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(_json_dump(report), encoding="utf-8")
    with connect() as conn:
        conn.execute(
            """
            UPDATE lora_comparison_sessions
            SET report_md_path = ?, report_json_path = ?, updated_at = ?
            WHERE id = ?
            """,
            (str(md_path), str(json_path), utc_now(), session_id),
        )
    return str(md_path), str(json_path)


def archive_lora_comparison_session(session_id: int) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE lora_comparison_sessions SET status = 'archived', updated_at = ? WHERE id = ?",
            (utc_now(), session_id),
        )
