from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image, ImageOps

from app.db import connect, fetch_all, fetch_one, utc_now
from app.services.embedding_service import (
    EmbeddingSource,
    embedding_cache_path,
    image_metadata,
    sources_for_job_type,
    upsert_image_embedding,
)


def mock_embedding(image_path: Path, vector_dim: int = 512, normalize: bool = True) -> np.ndarray:
    digest = hashlib.sha256(image_path.read_bytes()).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(vector_dim).astype(np.float32)
    if normalize:
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
    return vector


def resolve_torch_device_and_dtype(torch_module: Any, requested_device: str, requested_dtype: str) -> tuple[Any, Any, str, str]:
    """Resolve device/dtype for real embedding providers without forcing CUDA."""
    requested_device = (requested_device or "auto").lower()
    requested_dtype = (requested_dtype or "fp32").lower()
    cuda_available = bool(getattr(torch_module.cuda, "is_available", lambda: False)())
    if requested_device == "cuda" and not cuda_available:
        resolved_device = "cpu"
    elif requested_device == "cuda":
        resolved_device = "cuda"
    elif requested_device == "cpu":
        resolved_device = "cpu"
    else:
        resolved_device = "cuda" if cuda_available else "cpu"

    if resolved_device == "cpu":
        resolved_dtype = "fp32"
        torch_dtype = getattr(torch_module, "float32")
    elif requested_dtype == "fp16":
        resolved_dtype = "fp16"
        torch_dtype = getattr(torch_module, "float16")
    elif requested_dtype == "bf16" and hasattr(torch_module, "bfloat16"):
        resolved_dtype = "bf16"
        torch_dtype = getattr(torch_module, "bfloat16")
    else:
        resolved_dtype = "fp32"
        torch_dtype = getattr(torch_module, "float32")
    return torch_module.device(resolved_device), torch_dtype, resolved_device, resolved_dtype


class TransformersClipProvider:
    def __init__(self, model: dict[str, Any], settings: dict[str, Any]) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:  # pragma: no cover - depends on optional runtime packages
            raise RuntimeError(
                "transformers_clip providerを使うには torch と transformers が必要です。"
            ) from exc

        self.torch = torch
        self.normalize = bool(model.get("normalize"))
        self.device, self.torch_dtype, self.device_name, self.dtype_name = resolve_torch_device_and_dtype(
            torch,
            str(settings.get("device") or "auto"),
            str(settings.get("dtype") or "fp32"),
        )
        self.model_name = str(model.get("model_name") or "openai/clip-vit-base-patch32")
        local_files_only = not bool(settings.get("allow_model_download"))
        try:
            self.processor = CLIPProcessor.from_pretrained(self.model_name, local_files_only=local_files_only)
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                local_files_only=local_files_only,
                torch_dtype=self.torch_dtype,
            )
        except Exception as exc:
            if local_files_only:
                raise RuntimeError(
                    f"{self.model_name} がローカルに見つかりません。Embedding設定で allow_model_download=true "
                    "（モデルdownloadを許可）にしてから再実行してください。"
                ) from exc
            raise
        self.model.to(self.device)
        self.model.eval()

    def embedding(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return self.embedding_from_image(image)

    def embedding_from_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(device=self.device, dtype=self.torch_dtype) if getattr(value, "is_floating_point", lambda: False)() else value.to(self.device)
            for key, value in inputs.items()
        }
        with self.torch.no_grad():
            features = self.model.get_image_features(**inputs)
        vector = features[0].detach().float().cpu().numpy().astype(np.float32)
        if self.normalize:
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
        return vector


class OpenClipProvider:
    def __init__(self, model: dict[str, Any], settings: dict[str, Any]) -> None:
        try:
            import torch
            import open_clip
        except Exception as exc:  # pragma: no cover - depends on optional runtime packages
            raise RuntimeError(
                "open_clip providerを使うには torch と open_clip_torch が必要です。"
            ) from exc

        self.torch = torch
        self.open_clip = open_clip
        self.normalize = bool(model.get("normalize"))
        self.device, self.torch_dtype, self.device_name, self.dtype_name = resolve_torch_device_and_dtype(
            torch,
            str(settings.get("device") or "auto"),
            str(settings.get("dtype") or "fp16"),
        )
        self.model_name = str(model.get("model_name") or "ViT-B-32")
        self.pretrained = str(model.get("pretrained") or "laion2b_s34b_b79k")
        if not bool(settings.get("allow_model_download")) and not self._has_local_pretrained():
            raise RuntimeError(
                f"{self.model_name} / {self.pretrained} がローカルcacheに見つかりません。Embedding設定で "
                "allow_model_download=true（モデルdownloadを許可）にしてから再実行してください。"
            )
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device,
            )
        except Exception as exc:
            if not bool(settings.get("allow_model_download")):
                raise RuntimeError(
                    f"{self.model_name} / {self.pretrained} がローカルcacheに見つかりません。Embedding設定で "
                    "allow_model_download=true（モデルdownloadを許可）にしてから再実行してください。"
                ) from exc
            raise
        self.model.to(self.device)
        if self.device_name != "cpu" and self.dtype_name in {"fp16", "bf16"}:
            self.model.to(dtype=self.torch_dtype)
        self.model.eval()

    def _has_local_pretrained(self) -> bool:
        try:
            download_pretrained = getattr(self.open_clip, "download_pretrained", None)
            get_pretrained_cfg = getattr(self.open_clip, "get_pretrained_cfg", None)
            if download_pretrained is None or get_pretrained_cfg is None:
                return False
            cfg = get_pretrained_cfg(self.model_name, self.pretrained)
            if not cfg:
                return False
            path = download_pretrained(cfg, prefer_hf_hub=True, local_files_only=True)
            return bool(path and Path(path).exists())
        except Exception:
            return False

    def embedding(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return self.embedding_from_image(image)

    def embedding_from_image(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=self.torch_dtype)
        with self.torch.no_grad():
            features = self.model.encode_image(tensor)
        vector = features[0].detach().float().cpu().numpy().astype(np.float32)
        if self.normalize:
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
        return vector


def run_transformers_clip_preflight(model: dict[str, Any], settings_row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "provider": "transformers_clip",
        "model_id": model.get("id"),
        "model_name": model.get("model_name") or "openai/clip-vit-base-patch32",
        "status": "OK",
        "checks": [],
        "download_allowed": bool(settings_row.get("allow_model_download")),
    }
    try:
        import torch
        import transformers

        result["checks"].append({"name": "torch import", "status": "OK", "message": f"torch {torch.__version__}"})
        result["checks"].append({"name": "transformers import", "status": "OK", "message": f"transformers {transformers.__version__}"})
        device, torch_dtype, device_name, dtype_name = resolve_torch_device_and_dtype(
            torch,
            str(settings_row.get("device") or "auto"),
            str(settings_row.get("dtype") or "fp32"),
        )
        cuda_available = bool(torch.cuda.is_available())
        if device_name == "cpu" and str(settings_row.get("device") or "auto").lower() == "cuda":
            result["status"] = "WARNING"
        result["checks"].append(
            {
                "name": "device",
                "status": "OK" if device_name == "cuda" or str(settings_row.get("device") or "auto").lower() != "cuda" else "WARNING",
                "message": f"cuda_available={cuda_available}; selected_device={device_name}; selected_dtype={dtype_name}",
            }
        )
        provider = TransformersClipProvider(model, settings_row)
        result["checks"].append({"name": "model load", "status": "OK", "message": provider.model_name})
        image = Image.new("RGB", (32, 32), color=(128, 128, 128))
        vector = provider.embedding_from_image(image)
        norm = float(np.linalg.norm(vector))
        result["vector_dim"] = int(vector.shape[0])
        result["norm"] = norm
        result["device"] = provider.device_name
        result["dtype"] = provider.dtype_name
        result["checks"].append(
            {
                "name": "dummy image embedding",
                "status": "OK" if int(vector.shape[0]) == 512 else "WARNING",
                "message": f"vector_dim={int(vector.shape[0])}; norm={norm:.6f}",
            }
        )
        if int(vector.shape[0]) != 512:
            result["status"] = "WARNING"
    except Exception as exc:
        result["status"] = "ERROR"
        result["checks"].append({"name": "transformers_clip preflight", "status": "ERROR", "message": str(exc)})
    return result


def run_open_clip_preflight(model: dict[str, Any], settings_row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "provider": "open_clip",
        "model_id": model.get("id"),
        "model_name": model.get("model_name") or "ViT-B-32",
        "pretrained": model.get("pretrained") or "laion2b_s34b_b79k",
        "status": "OK",
        "checks": [],
        "download_allowed": bool(settings_row.get("allow_model_download")),
    }
    try:
        import torch
        import open_clip

        version = getattr(open_clip, "__version__", "unknown")
        result["checks"].append({"name": "torch import", "status": "OK", "message": f"torch {torch.__version__}"})
        result["checks"].append({"name": "open_clip import", "status": "OK", "message": f"open_clip {version}"})
        _, _, device_name, dtype_name = resolve_torch_device_and_dtype(
            torch,
            str(settings_row.get("device") or "auto"),
            str(settings_row.get("dtype") or "fp16"),
        )
        cuda_available = bool(torch.cuda.is_available())
        if device_name == "cpu" and str(settings_row.get("device") or "auto").lower() == "cuda":
            result["status"] = "WARNING"
        result["checks"].append(
            {
                "name": "device",
                "status": "OK" if device_name == "cuda" or str(settings_row.get("device") or "auto").lower() != "cuda" else "WARNING",
                "message": f"cuda_available={cuda_available}; selected_device={device_name}; selected_dtype={dtype_name}",
            }
        )
        provider = OpenClipProvider(model, settings_row)
        result["checks"].append({"name": "model load", "status": "OK", "message": f"{provider.model_name} / {provider.pretrained}"})
        image = Image.new("RGB", (32, 32), color=(128, 128, 128))
        vector = provider.embedding_from_image(image)
        norm = float(np.linalg.norm(vector))
        result["vector_dim"] = int(vector.shape[0])
        result["norm"] = norm
        result["device"] = provider.device_name
        result["dtype"] = provider.dtype_name
        result["checks"].append(
            {
                "name": "dummy image embedding",
                "status": "OK" if int(vector.shape[0]) > 0 else "WARNING",
                "message": f"vector_dim={int(vector.shape[0])}; norm={norm:.6f}",
            }
        )
        if int(model.get("vector_dim") or vector.shape[0]) != int(vector.shape[0]):
            result["status"] = "WARNING"
            result["checks"].append(
                {
                    "name": "vector_dim",
                    "status": "WARNING",
                    "message": f"DB={model.get('vector_dim') or '-'}; actual={int(vector.shape[0])}",
                }
            )
    except Exception as exc:
        result["status"] = "ERROR"
        result["checks"].append({"name": "open_clip preflight", "status": "ERROR", "message": str(exc)})
    return result


def load_image_for_validation(path: Path, max_image_size: int) -> None:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if max_image_size and max(image.size) > max_image_size:
            image.thumbnail((max_image_size, max_image_size))


def source_from_item(item: Any, source_map: dict[tuple[str, int | None, str], EmbeddingSource]) -> EmbeddingSource:
    key = (item["source_type"], item["source_id"], item["source_path"])
    source = source_map.get(key)
    if source:
        return source
    return EmbeddingSource(item["source_type"], item["source_id"], item["source_path"])


def update_job_counts(conn: Any, job_id: int) -> None:
    counts = {
        row["status"]: row["count"]
        for row in conn.execute(
            "SELECT status, COUNT(*) AS count FROM embedding_job_items WHERE embedding_job_id = ? GROUP BY status",
            (job_id,),
        ).fetchall()
    }
    processed = sum(counts.get(status, 0) for status in ("ready", "failed", "skipped", "missing_source"))
    conn.execute(
        """
        UPDATE embedding_jobs
        SET processed_count = ?, ready_count = ?, failed_count = ?,
            skipped_count = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            processed,
            counts.get("ready", 0),
            counts.get("failed", 0) + counts.get("missing_source", 0),
            counts.get("skipped", 0),
            utc_now(),
            job_id,
        ),
    )


def run_embedding_job(job_id: int) -> int:
    job = fetch_one("SELECT * FROM embedding_jobs WHERE id = ?", (job_id,))
    if job is None:
        print(f"Embedding job #{job_id} not found", flush=True)
        return 2
    model = fetch_one("SELECT * FROM embedding_models WHERE id = ?", (job["embedding_model_id"],))
    settings = fetch_one("SELECT * FROM embedding_settings ORDER BY id LIMIT 1")
    if model is None or settings is None:
        print("Embedding model or settings not found", flush=True)
        return 2

    started = time.time()
    now = utc_now()
    with connect() as conn:
        conn.execute("UPDATE embedding_jobs SET status = 'running', started_at = COALESCE(started_at, ?), updated_at = ? WHERE id = ?", (now, now, job_id))

    provider = model["provider"]
    vector_dim = int(model["vector_dim"] or 512)
    normalize = bool(model["normalize"])
    max_image_size = int(settings["max_image_size"] or 1024)
    items = fetch_all("SELECT * FROM embedding_job_items WHERE embedding_job_id = ? ORDER BY id", (job_id,))
    vector_provider: TransformersClipProvider | OpenClipProvider | None = None
    try:
        if provider == "transformers_clip":
            vector_provider = TransformersClipProvider(dict(model), dict(settings))
            print(
                f"transformers_clip ready: model={vector_provider.model_name}, device={vector_provider.device_name}, dtype={vector_provider.dtype_name}, download_allowed={bool(settings['allow_model_download'])}",
                flush=True,
            )
        elif provider == "open_clip":
            vector_provider = OpenClipProvider(dict(model), dict(settings))
            print(
                f"open_clip ready: model={vector_provider.model_name}, pretrained={vector_provider.pretrained}, device={vector_provider.device_name}, dtype={vector_provider.dtype_name}, download_allowed={bool(settings['allow_model_download'])}",
                flush=True,
            )
        elif provider not in {"mock"}:
            raise RuntimeError(f"provider {provider} は実行未対応です。")
    except Exception as exc:
        error_message = str(exc)
        finished = utc_now()
        elapsed = int(time.time() - started)
        print(f"Embedding job #{job_id} failed during provider setup: {error_message}", flush=True)
        with connect() as conn:
            conn.execute(
                """
                UPDATE embedding_job_items
                SET status = 'failed', error_message = ?, updated_at = ?
                WHERE embedding_job_id = ? AND status IN ('pending', 'planned')
                """,
                (error_message, finished, job_id),
            )
            conn.execute(
                """
                UPDATE embedding_jobs
                SET status = 'failed', processed_count = total_count, failed_count = total_count,
                    ended_at = ?, elapsed_seconds = ?, return_code = 1,
                    updated_at = ?, error_message = ?
                WHERE id = ?
                """,
                (finished, elapsed, finished, error_message, job_id),
            )
        return 1
    source_map = {
        (source.source_type, source.source_id, source.source_path): source
        for source in sources_for_job_type(job["job_type"], job["target_id"])
    }
    print(f"Embedding job #{job_id} started: provider={provider}, items={len(items)}", flush=True)

    for index, item in enumerate(items, start=1):
        item_id = int(item["id"])
        path = Path(item["source_path"])
        status = "ready"
        error = ""
        embedding_id = None
        try:
            if not path.exists() or not path.is_file():
                raise FileNotFoundError("source image is missing")
            load_image_for_validation(path, max_image_size)
            metadata = image_metadata(path, max_image_size=max_image_size)
            source = source_from_item(item, source_map)
            if provider == "mock":
                vector = mock_embedding(path, vector_dim=vector_dim, normalize=normalize)
            elif provider in {"transformers_clip", "open_clip"} and vector_provider is not None:
                vector = vector_provider.embedding(path)
            else:
                raise RuntimeError(f"provider {provider} は実行未対応です。")
            metadata["vector_dim"] = int(vector.shape[0])
            target = embedding_cache_path(model["id"], item["source_type"], item["source_id"], item["source_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, vector)
            embedding_id = upsert_image_embedding(source, dict(model), metadata, target, "ready")
            print(f"[{index}/{len(items)}] ready {item['source_type']}#{item['source_id'] or '-'} {path.name}", flush=True)
        except FileNotFoundError as exc:
            status = "missing_source"
            error = str(exc)
            metadata = {}
            source = source_from_item(item, source_map)
            embedding_id = upsert_image_embedding(source, dict(model), metadata, Path(""), "missing_source", error)
            print(f"[{index}/{len(items)}] missing {path}: {error}", flush=True)
        except Exception as exc:
            status = "failed"
            error = str(exc)
            try:
                source = source_from_item(item, source_map)
                embedding_id = upsert_image_embedding(source, dict(model), {}, Path(""), "failed", error)
            except Exception:
                embedding_id = None
            print(f"[{index}/{len(items)}] failed {path}: {error}", flush=True)

        with connect() as conn:
            conn.execute(
                """
                UPDATE embedding_job_items
                SET status = ?, embedding_id = ?, error_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, embedding_id, error, utc_now(), item_id),
            )
            update_job_counts(conn, job_id)

    finished = utc_now()
    elapsed = int(time.time() - started)
    with connect() as conn:
        counts = conn.execute("SELECT failed_count FROM embedding_jobs WHERE id = ?", (job_id,)).fetchone()
        failed = int(counts["failed_count"] or 0) if counts else 0
        final_status = "failed" if failed and failed == len(items) else "completed"
        conn.execute(
            """
            UPDATE embedding_jobs
            SET status = ?, ended_at = ?, elapsed_seconds = ?, return_code = ?,
                updated_at = ?, error_message = ?
            WHERE id = ?
            """,
            (final_status, finished, elapsed, 0 if final_status == "completed" else 1, finished, "" if final_status == "completed" else "all items failed", job_id),
        )
    print(f"Embedding job #{job_id} finished: status={final_status}, elapsed={elapsed}s", flush=True)
    return 0 if final_status == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-job-id", type=int)
    parser.add_argument("--preflight-model-id")
    args = parser.parse_args()
    if args.preflight_model_id:
        model = fetch_one("SELECT * FROM embedding_models WHERE id = ?", (args.preflight_model_id,))
        settings_row = fetch_one("SELECT * FROM embedding_settings ORDER BY id LIMIT 1")
        if model is None or settings_row is None:
            print(json.dumps({"status": "ERROR", "checks": [{"name": "db", "status": "ERROR", "message": "model or settings not found"}]}, ensure_ascii=False))
            return 2
        if model["provider"] == "transformers_clip":
            print(json.dumps(run_transformers_clip_preflight(dict(model), dict(settings_row)), ensure_ascii=False))
            return 0
        if model["provider"] == "open_clip":
            print(json.dumps(run_open_clip_preflight(dict(model), dict(settings_row)), ensure_ascii=False))
            return 0
        print(json.dumps({"status": "ERROR", "checks": [{"name": "provider", "status": "ERROR", "message": f"unsupported provider: {model['provider']}"}]}, ensure_ascii=False))
        return 2
    if args.embedding_job_id is None:
        parser.error("--embedding-job-id or --preflight-model-id is required")
    return run_embedding_job(args.embedding_job_id)


if __name__ == "__main__":
    raise SystemExit(main())
