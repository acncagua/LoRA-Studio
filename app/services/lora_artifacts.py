from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.db import fetch_one
from app.services.output_collector import safe_sha256_file


@dataclass(frozen=True)
class ResolvedLoraArtifact:
    path: Path
    source_kind: str
    expected_sha256: str | None
    actual_sha256: str
    file_size: int
    verified: bool
    warnings: tuple[str, ...]
    profile_id: int | None = None
    output_id: int | None = None
    job_id: int | None = None


def resolve_lora_artifact(
    *,
    profile_id: int | None = None,
    output_id: int | None = None,
    job_id: int | None = None,
) -> ResolvedLoraArtifact:
    profile = fetch_one("SELECT * FROM selected_lora_profiles WHERE id = ?", (profile_id,)) if profile_id else None
    output = fetch_one("SELECT * FROM training_outputs WHERE id = ?", (output_id,)) if output_id else None
    if output is None and profile is not None and profile["selected_output_id"]:
        output = fetch_one("SELECT * FROM training_outputs WHERE id = ?", (profile["selected_output_id"],))
    if output is None and job_id is not None:
        output = fetch_one("SELECT * FROM training_outputs WHERE job_id = ? AND selected = 1 ORDER BY id DESC LIMIT 1", (job_id,))
    if profile is None and output is not None:
        profile = fetch_one(
            "SELECT * FROM selected_lora_profiles WHERE selected_output_id = ? ORDER BY updated_at DESC, id DESC LIMIT 1",
            (output["id"],),
        )
    job = fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id or (output["job_id"] if output else None) or (profile["job_id"] if profile else None),))
    expected_sha = (output["sha256"] if output and "sha256" in output.keys() else None) or None

    candidates: list[tuple[str, str | None, str | None]] = []
    if output is not None:
        candidates.append(("training_outputs.file_path", output["file_path"]))
        candidates.append(("training_outputs.external_copy_path", output["external_copy_path"] if "external_copy_path" in output.keys() else None))
    if profile is not None:
        candidates.append(("selected_lora_profiles.exported_model_path", profile["exported_model_path"] if "exported_model_path" in profile.keys() else None))
        candidates.append(("selected_lora_profiles.selected_model_path", profile["selected_model_path"] if "selected_model_path" in profile.keys() else None))
    if job is not None:
        candidates.append(("training_jobs.adopted_model_path", job["adopted_model_path"] if "adopted_model_path" in job.keys() else None))

    warnings: list[str] = []
    seen: set[str] = set()
    rejected: list[str] = []
    for source_kind, value in candidates:
        if not value:
            continue
        path = Path(value)
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if not path.exists() or not path.is_file():
            rejected.append(f"{source_kind}: missing")
            continue
        actual_sha, error = safe_sha256_file(path)
        if error or not actual_sha:
            rejected.append(f"{source_kind}: sha256 error {error or ''}".strip())
            continue
        if expected_sha and actual_sha != expected_sha:
            rejected.append(f"{source_kind}: sha256 mismatch")
            continue
        if not expected_sha:
            warnings.append("expected_sha256_not_recorded")
        return ResolvedLoraArtifact(
            path=path,
            source_kind=source_kind,
            expected_sha256=expected_sha,
            actual_sha256=actual_sha,
            file_size=path.stat().st_size,
            verified=True,
            warnings=tuple(warnings),
            profile_id=int(profile["id"]) if profile is not None else profile_id,
            output_id=int(output["id"]) if output is not None else output_id,
            job_id=int(job["id"]) if job is not None else job_id,
        )

    detail = "; ".join(rejected) if rejected else "no candidate paths"
    raise ValueError(f"LoRA artifact could not be resolved ({detail})")


def resolve_lora_artifact_for_validation_run(run: Any) -> ResolvedLoraArtifact:
    return resolve_lora_artifact(
        profile_id=run["selected_lora_profile_id"] if "selected_lora_profile_id" in run.keys() else None,
        output_id=run["selected_output_id"] if "selected_output_id" in run.keys() else None,
        job_id=run["job_id"] if "job_id" in run.keys() else None,
    )
