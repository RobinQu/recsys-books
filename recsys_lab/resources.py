"""Idempotent, manifest-driven access to local papers and full datasets."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "config" / "resources.json"
RESOURCE_ROOT = ROOT / "resources"
LOCK_PATH = RESOURCE_ROOT / "resource-lock.json"
USER_AGENT = "RecSysAtlas/1.0 (educational resource initializer; contact via repository)"


@dataclass
class ResourceState:
    id: str
    kind: str
    path: str
    status: str
    size: int = 0
    sha256: str | None = None
    message: str | None = None


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def iter_resources(kinds: Iterable[str] | None = None, ids: Iterable[str] | None = None):
    manifest = load_manifest()
    wanted_kinds = set(kinds or manifest.keys())
    wanted_ids = set(ids or [])
    for kind in ("papers", "datasets", "vendor"):
        if kind not in wanted_kinds:
            continue
        for item in manifest.get(kind, []):
            if not wanted_ids or item["id"] in wanted_ids:
                yield kind, item


def target_path(item: dict) -> Path:
    return RESOURCE_ROOT / item["path"]


def source_url(item: dict) -> str | None:
    if item["provider"] == "arxiv":
        return f"https://arxiv.org/pdf/{item['arxiv_id']}.pdf"
    return item.get("url")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_resource(kind: str, item: dict) -> ResourceState:
    path = target_path(item)
    if not path.exists():
        return ResourceState(item["id"], kind, str(path.relative_to(ROOT)), "missing")
    if item["provider"] in {"huggingface", "npm"}:
        files = [entry for entry in path.rglob("*") if entry.is_file() and ".cache" not in entry.parts]
        minimum = 2 if item["provider"] == "npm" else 1
        by_name = {entry.name: entry for entry in files}
        missing = sorted(set(item.get("expected_files", [])) - set(by_name))
        if len(files) < minimum or missing:
            message = f"missing expected files: {', '.join(missing)}" if missing else "resource directory is incomplete"
            return ResourceState(item["id"], kind, str(path.relative_to(ROOT)), "missing", sum(f.stat().st_size for f in files), message=message)
        for name, expected_hash in item.get("file_sha256", {}).items():
            if name not in by_name or _sha256(by_name[name]) != expected_hash:
                return ResourceState(item["id"], kind, str(path.relative_to(ROOT)), "invalid", sum(f.stat().st_size for f in files), message=f"sha256 mismatch: {name}")
        tree_digest = hashlib.sha256()
        for entry in sorted(files):
            tree_digest.update(str(entry.relative_to(path)).encode())
            tree_digest.update(_sha256(entry).encode())
        return ResourceState(item["id"], kind, str(path.relative_to(ROOT)), "ready", sum(f.stat().st_size for f in files), tree_digest.hexdigest())
    if not path.is_file() or path.stat().st_size == 0:
        return ResourceState(item["id"], kind, str(path.relative_to(ROOT)), "invalid", message="empty resource")
    if path.stat().st_size < item.get("min_size_bytes", 1):
        return ResourceState(
            item["id"], kind, str(path.relative_to(ROOT)), "invalid", path.stat().st_size,
            message=f"resource is smaller than min_size_bytes={item['min_size_bytes']}",
        )
    digest = _sha256(path)
    expected = item.get("sha256")
    if expected and digest != expected:
        return ResourceState(item["id"], kind, str(path.relative_to(ROOT)), "invalid", path.stat().st_size, digest, "sha256 mismatch")
    if kind == "papers":
        with path.open("rb") as handle:
            if handle.read(4) != b"%PDF":
                return ResourceState(item["id"], kind, str(path.relative_to(ROOT)), "invalid", path.stat().st_size, digest, "not a PDF")
        # When pypdf is installed (the application image includes it), verify
        # that a valid-looking PDF is actually the paper named in the manifest.
        # The root initializer remains usable in a standard-library-only Python.
        try:
            from pypdf import PdfReader

            logger = logging.getLogger("pypdf")
            previous_level = logger.level
            logger.setLevel(logging.ERROR)
            try:
                reader = PdfReader(path)
                sample = " ".join((page.extract_text() or "") for page in reader.pages[:2]).casefold()
            finally:
                logger.setLevel(previous_level)
            significant = [
                token.strip("():,-").casefold()
                for token in item["title"].split()
                if len(token.strip("():,-")) >= 5
            ]
            matches = sum(token in sample for token in significant)
            if significant and matches / len(significant) < 0.55:
                return ResourceState(
                    item["id"], kind, str(path.relative_to(ROOT)), "invalid",
                    path.stat().st_size, digest, "PDF title does not match manifest",
                )
        except ImportError:
            pass
        except Exception as error:
            return ResourceState(item["id"], kind, str(path.relative_to(ROOT)), "invalid", path.stat().st_size, digest, f"PDF text validation failed: {error}")
    if path.name.endswith(".tar.gz"):
        try:
            with tarfile.open(path, "r:gz") as bundle:
                names = [Path(member.name).name for member in bundle.getmembers() if member.isfile()]
            missing_members = sorted(set(item.get("expected_members", [])) - set(names))
            if missing_members:
                raise ValueError(f"missing archive members: {', '.join(missing_members)}")
        except (tarfile.TarError, EOFError, OSError, ValueError) as error:
            return ResourceState(item["id"], kind, str(path.relative_to(ROOT)), "invalid", path.stat().st_size, digest, f"invalid archive: {error}")
    if path.name.endswith(".zip"):
        try:
            with zipfile.ZipFile(path) as bundle:
                names = [name for name in bundle.namelist() if not name.endswith("/")]
            available = set(names) | {Path(name).name for name in names}
            missing_members = sorted(member for member in item.get("expected_members", []) if member not in available)
            if missing_members:
                raise ValueError(f"missing archive members: {', '.join(missing_members)}")
        except (zipfile.BadZipFile, OSError, ValueError) as error:
            return ResourceState(item["id"], kind, str(path.relative_to(ROOT)), "invalid", path.stat().st_size, digest, f"invalid archive: {error}")
    if path.name.endswith((".json.gz", ".csv.gz")):
        try:
            import gzip
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                first_line = handle.readline()
            if not first_line.strip():
                raise ValueError("compressed dataset has no rows")
            for column in item.get("required_columns", []):
                if column not in first_line:
                    raise ValueError(f"missing required column marker: {column}")
        except (OSError, UnicodeError, ValueError) as error:
            return ResourceState(item["id"], kind, str(path.relative_to(ROOT)), "invalid", path.stat().st_size, digest, f"invalid compressed dataset: {error}")
    if path.suffix == ".csv" and item.get("first_row_fields"):
        try:
            with path.open("rt", encoding="utf-8") as handle:
                fields = handle.readline().rstrip("\r\n").split(",")
            if len(fields) != int(item["first_row_fields"]):
                raise ValueError(f"expected {item['first_row_fields']} fields, found {len(fields)}")
        except (OSError, UnicodeError, ValueError) as error:
            return ResourceState(item["id"], kind, str(path.relative_to(ROOT)), "invalid", path.stat().st_size, digest, f"invalid CSV dataset: {error}")
    return ResourceState(item["id"], kind, str(path.relative_to(ROOT)), "ready", path.stat().st_size, digest)


def _download_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            temporary.replace(destination)
            return
        except Exception as error:  # network failures are reported in the lock
            last_error = error
            temporary.unlink(missing_ok=True)
            time.sleep(2**attempt)
    raise RuntimeError(f"download failed: {url}: {last_error}")


def _download_huggingface(item: dict, destination: Path, offline: bool) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("Hugging Face resources require huggingface_hub (installed in the project image)") from error
    token = os.getenv("HF_TOKEN")
    snapshot_download(
        repo_id=item["repo_id"],
        repo_type=item.get("repo_type", "dataset"),
        revision=item.get("revision"),
        local_dir=destination,
        allow_patterns=item.get("allow_patterns"),
        token=token,
        local_files_only=offline,
    )


def _download_npm(item: dict, destination: Path) -> None:
    encoded = urllib.parse.quote(item["package"], safe="")
    metadata_url = f"https://registry.npmjs.org/{encoded}/{item['version']}"
    request = urllib.request.Request(metadata_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        metadata = json.load(response)
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "package.tgz"
        _download_url(metadata["dist"]["tarball"], archive)
        extract_root = Path(temporary) / "extract"
        with tarfile.open(archive, "r:gz") as bundle:
            safe_members = [member for member in bundle.getmembers() if member.name.startswith("package/dist/") and ".." not in Path(member.name).parts]
            bundle.extractall(extract_root, members=safe_members, filter="data")
        source = extract_root / "package" / "dist"
        if not source.exists():
            raise RuntimeError("npm package does not contain dist/")
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(source, destination)


def ensure_resources(
    *, download: bool = False, strict: bool = False, offline: bool = False,
    kinds: Iterable[str] | None = None, ids: Iterable[str] | None = None,
    include_optional: bool = False,
) -> dict:
    """Verify resources and optionally download missing manifest entries."""
    RESOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    states: list[ResourceState] = []
    fetched = 0  # resources actually downloaded this run (ready items are skipped)
    for kind, item in iter_resources(kinds, ids):
        state = inspect_resource(kind, item)
        should_fetch = download and state.status != "ready" and (item.get("required", False) or include_optional)
        if should_fetch and not offline:
            try:
                if item["provider"] == "huggingface":
                    _download_huggingface(item, target_path(item), offline)
                elif item["provider"] == "npm":
                    _download_npm(item, target_path(item))
                else:
                    url = source_url(item)
                    if not url:
                        raise RuntimeError("no direct download URL; Scholar is discovery metadata only")
                    _download_url(url, target_path(item))
                state = inspect_resource(kind, item)
                if state.status == "ready":
                    fetched += 1
            except Exception as error:
                state.message = str(error)
        states.append(state)
    # 合并历史锁文件：按 --id/--kind 过滤的调用只更新涉及条目，
    # 不能整文件覆写，否则单次小范围调用会丢失其他资源的就绪记录。
    merged: dict[str, dict] = {}
    if LOCK_PATH.exists():
        try:
            merged = {
                item["id"]: item
                for item in json.loads(LOCK_PATH.read_text(encoding="utf-8")).get("items", [])
            }
        except json.JSONDecodeError:
            merged = {}
    merged.update({state.id: asdict(state) for state in states})
    payload = {
        "schema_version": 1,
        "generated_at": int(time.time()),
        "ready": sum(item["status"] == "ready" for item in merged.values()),
        "missing": sum(item["status"] != "ready" for item in merged.values()),
        "fetched": fetched,
        "items": list(merged.values()),
    }
    LOCK_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Explicitly requested IDs are an execution contract even when the same
    # resource is optional during a normal application startup.
    requested = set(ids or [])
    required = {
        item["id"] for _, item in iter_resources(kinds, ids)
        if item.get("required", False) or item["id"] in requested
    }
    failed_required = [state for state in states if state.id in required and state.status != "ready"]
    if strict and failed_required:
        details = ", ".join(f"{state.id}: {state.message or state.status}" for state in failed_required)
        raise RuntimeError(f"required resources are not ready: {details}")
    return payload


def paper_catalog() -> dict[str, dict]:
    return {item["id"]: item for item in load_manifest()["papers"]}
