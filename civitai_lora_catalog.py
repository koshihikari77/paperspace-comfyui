from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests
import yaml


CIVITAI_BASE_URL = "https://civitai.com"
DEFAULT_OUTPUT = Path("workspace/lora_catalog/civitai_loras.yaml")
DEFAULT_JSONL_OUTPUT = Path("workspace/lora_catalog/civitai_loras.jsonl")
DEFAULT_RAW_OUTPUT = Path("workspace/lora_catalog/civitai_loras.raw_versions.yaml")
DEFAULT_RAW_JSONL_OUTPUT = Path("workspace/lora_catalog/civitai_loras.raw_versions.jsonl")
DEFAULT_DOWNLOAD_DIR = Path("workspace/lora_catalog/downloads")


@dataclass(frozen=True)
class CollectionItemRef:
    model_id: int
    note: str | None = None
    collection_item_id: int | None = None


class CivitaiCatalogError(RuntimeError):
    pass


def _clean_html(value: str | None) -> str:
    if not value:
        return ""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(value, "html.parser")
    text = soup.get_text("\n")
    text = html.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9._-]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered).strip("-._")
    return lowered or "lora"


def _collection_id_from_value(value: str) -> int:
    if value.isdigit():
        return int(value)
    parsed = urlparse(value)
    parts = [part for part in parsed.path.split("/") if part]
    for index, part in enumerate(parts):
        if part == "collections" and index + 1 < len(parts):
            candidate = parts[index + 1]
            if candidate.isdigit():
                return int(candidate)
        if part.isdigit():
            return int(part)
    raise CivitaiCatalogError(f"collection id を URL から抽出できません: {value}")


def _base_url_from_collection(value: str | None) -> str:
    if not value:
        return CIVITAI_BASE_URL
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return CIVITAI_BASE_URL


def _request_json(
    session: requests.Session,
    url: str,
    *,
    token: str | None,
    params: dict[str, Any] | None = None,
) -> Any:
    headers = {
        "Accept": "application/json",
        "User-Agent": "comfyv-lora-catalog/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = session.get(url, headers=headers, params=params, timeout=60)
    if response.status_code >= 400:
        raise CivitaiCatalogError(f"GET {response.url} failed: {response.status_code} {response.text[:500]}")
    return response.json()


def _trpc_get(
    session: requests.Session,
    procedure: str,
    payload: dict[str, Any],
    *,
    token: str | None,
    base_url: str = CIVITAI_BASE_URL,
) -> Any:
    headers = {
        "Accept": "application/json",
        "User-Agent": "comfyv-lora-catalog/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    encoded_input = quote(json.dumps({"json": payload}, separators=(",", ":")))
    url = f"{base_url}/api/trpc/{procedure}?input={encoded_input}"
    response = session.get(url, headers=headers, timeout=60)
    if response.status_code >= 400:
        raise CivitaiCatalogError(f"GET {response.url} failed: {response.status_code} {response.text[:500]}")
    raw = response.json()
    return raw.get("result", {}).get("data", {}).get("json", raw)


def fetch_collection_model_refs(
    session: requests.Session,
    collection_id: int,
    *,
    token: str | None,
    base_url: str = CIVITAI_BASE_URL,
    limit: int = 100,
) -> list[CollectionItemRef]:
    refs: list[CollectionItemRef] = []
    cursor: str | None = None
    seen_items: set[int] = set()

    while True:
        payload: dict[str, Any] = {
            "collectionId": collection_id,
            "limit": limit,
            "statuses": ["ACCEPTED"],
        }
        if cursor:
            payload["cursor"] = cursor

        data = _trpc_get(
            session,
            "collection.getAllCollectionItems",
            payload,
            token=token,
            base_url=base_url,
        )
        items = data.get("collectionItems") or data.get("items") or []
        for item in items:
            collection_item_id = item.get("id")
            if isinstance(collection_item_id, int) and collection_item_id in seen_items:
                continue
            if isinstance(collection_item_id, int):
                seen_items.add(collection_item_id)

            if item.get("type") != "model":
                continue
            model = item.get("data") or {}
            model_id = model.get("id") or item.get("modelId")
            if isinstance(model_id, int):
                refs.append(
                    CollectionItemRef(
                        model_id=model_id,
                        note=item.get("note"),
                        collection_item_id=collection_item_id if isinstance(collection_item_id, int) else None,
                    )
                )

        next_cursor = data.get("nextCursor")
        if not next_cursor:
            break
        cursor = str(next_cursor)

    deduped: dict[int, CollectionItemRef] = {}
    for ref in refs:
        deduped[ref.model_id] = ref
    return list(deduped.values())


def fetch_model(
    session: requests.Session,
    model_id: int,
    *,
    token: str | None,
    base_url: str = CIVITAI_BASE_URL,
) -> dict[str, Any]:
    params = {"token": token} if token else None
    return _request_json(session, f"{base_url}/api/v1/models/{model_id}", token=token, params=params)


def _primary_model_file(version: dict[str, Any]) -> dict[str, Any] | None:
    files = version.get("files") or []
    model_files = [file for file in files if file.get("type") == "Model"]
    candidates = model_files or files
    for file in candidates:
        if file.get("primary"):
            return file
    return candidates[0] if candidates else None


def _select_version(model: dict[str, Any], preferred_version_id: int | None) -> dict[str, Any] | None:
    versions = model.get("modelVersions") or []
    if preferred_version_id is not None:
        for version in versions:
            if version.get("id") == preferred_version_id:
                return version
    for version in versions:
        files = version.get("files") or []
        if any(file.get("primary") and file.get("type") == "Model" for file in files):
            return version
    return versions[0] if versions else None


def _normalize_tags(raw_tags: Any) -> list[str]:
    tags: list[str] = []
    if not isinstance(raw_tags, list):
        return tags
    for tag in raw_tags:
        if isinstance(tag, str) and tag:
            tags.append(tag)
        elif isinstance(tag, dict) and tag.get("name"):
            tags.append(str(tag["name"]))
    return tags


def normalize_lora_entry(
    model: dict[str, Any],
    *,
    ref: CollectionItemRef | None = None,
    preferred_version_id: int | None = None,
    version: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    version = version or _select_version(model, preferred_version_id)
    if not version:
        return None
    model_file = _primary_model_file(version)

    model_id = int(model["id"])
    version_id = int(version["id"])
    name = str(model.get("name") or f"model-{model_id}")
    version_name = str(version.get("name") or "")
    file_name = str(model_file.get("name")) if model_file else None
    key_source = file_name or f"{name}-{version_name}-{version_id}"

    creator = model.get("creator") or {}
    stats = model.get("stats") or {}
    version_stats = version.get("stats") or {}
    file_hashes = (model_file or {}).get("hashes") or {}

    entry = {
        "key": _slugify(Path(key_source).stem),
        "name": name,
        "model_id": model_id,
        "model_version_id": version_id,
        "version_name": version_name,
        "version_index": version.get("index"),
        "version_created_at": version.get("createdAt"),
        "version_published_at": version.get("publishedAt"),
        "type": model.get("type"),
        "base_model": version.get("baseModel"),
        "file_name": file_name,
        "file_id": (model_file or {}).get("id"),
        "file_size_kb": (model_file or {}).get("sizeKB"),
        "hashes": {
            key: file_hashes[key]
            for key in ("SHA256", "AutoV2", "AutoV1", "CRC32", "BLAKE3")
            if file_hashes.get(key)
        },
        "download_url": (model_file or {}).get("downloadUrl")
        or f"{CIVITAI_BASE_URL}/api/download/models/{version_id}",
        "page_url": f"{CIVITAI_BASE_URL}/models/{model_id}",
        "creator": {
            "username": creator.get("username"),
            "image": creator.get("image"),
        },
        "trigger_words": list(version.get("trainedWords") or []),
        "tags": _normalize_tags(model.get("tags")),
        "nsfw": model.get("nsfw"),
        "poi": model.get("poi"),
        "stats": {
            "download_count": stats.get("downloadCount"),
            "favorite_count": stats.get("favoriteCount"),
            "rating": stats.get("rating"),
            "version_download_count": version_stats.get("downloadCount"),
        },
        "description": _clean_html(model.get("description")),
        "version_description": _clean_html(version.get("description")),
        "collection": {
            "item_id": ref.collection_item_id if ref else None,
            "note": ref.note if ref else None,
        },
        "agent_notes": {
            "aliases": [],
            "recommended_weight": None,
            "usage_notes": "",
        },
    }
    return entry


def normalize_lora_version_entries(
    model: dict[str, Any],
    *,
    ref: CollectionItemRef | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for version in model.get("modelVersions") or []:
        entry = normalize_lora_entry(model, ref=ref, version=version)
        if entry is not None:
            entries.append(entry)
    return entries


def load_model_ids_file(path: Path) -> list[int]:
    values: list[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.search(r"/models/(\d+)|^(\d+)$", line)
        if not match:
            raise CivitaiCatalogError(f"model id として読めません: {line}")
        values.append(int(match.group(1) or match.group(2)))
    return values


def write_catalog(catalog: dict[str, Any], output: Path, jsonl_output: Path | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(catalog, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
    if jsonl_output:
        jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in catalog["items"]]
        jsonl_output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _download_file(
    session: requests.Session,
    item: dict[str, Any],
    *,
    token: str | None,
    output_dir: Path,
    skip_existing: bool = True,
) -> str:
    file_name = item.get("file_name")
    download_url = item.get("download_url")
    if not file_name or not download_url:
        return "skipped:no-file"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / str(file_name)
    expected_sha256 = ((item.get("hashes") or {}).get("SHA256") or "").upper()

    if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
        if not expected_sha256 or _sha256(output_path) == expected_sha256:
            return "skipped:exists"
        output_path.unlink()

    headers = {
        "Accept": "application/octet-stream",
        "User-Agent": "comfyv-lora-catalog/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    with session.get(str(download_url), headers=headers, stream=True, timeout=120) as response:
        if response.status_code >= 400:
            raise CivitaiCatalogError(
                f"download failed: {download_url} {response.status_code} {response.text[:500]}"
            )
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    if expected_sha256:
        actual = _sha256(tmp_path)
        if actual != expected_sha256:
            tmp_path.unlink(missing_ok=True)
            raise CivitaiCatalogError(
                f"SHA256 mismatch for {file_name}: expected {expected_sha256}, got {actual}"
            )

    tmp_path.replace(output_path)
    return "downloaded"


def download_catalog(args: argparse.Namespace) -> dict[str, int]:
    token = args.token or os.getenv("CIVITAI_API_TOKEN")
    catalog_path = Path(args.download_catalog)
    data = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    items = data.get("items") or []
    session = requests.Session()
    counts: dict[str, int] = {}

    for index, item in enumerate(items, start=1):
        file_name = item.get("file_name") or f"item-{index}"
        print(f"[{index}/{len(items)}] {file_name}", file=sys.stderr)
        status = _download_file(
            session,
            item,
            token=token,
            output_dir=args.download_dir,
            skip_existing=not args.force_download,
        )
        counts[status] = counts.get(status, 0) + 1
        print(f"  {status}", file=sys.stderr)

    return counts


def build_catalog(args: argparse.Namespace) -> dict[str, Any]:
    token = args.token or os.getenv("CIVITAI_API_TOKEN")
    session = requests.Session()
    collection_base_url = _base_url_from_collection(args.collection)

    refs: list[CollectionItemRef]
    source: dict[str, Any]
    if args.collection:
        collection_id = _collection_id_from_value(args.collection)
        refs = fetch_collection_model_refs(
            session,
            collection_id,
            token=token,
            base_url=collection_base_url,
        )
        source = {
            "type": "civitai_collection",
            "collection_id": collection_id,
            "base_url": collection_base_url,
        }
    elif args.model_ids_file:
        model_ids = load_model_ids_file(Path(args.model_ids_file))
        refs = [CollectionItemRef(model_id=model_id) for model_id in model_ids]
        source = {"type": "model_ids_file", "path": str(args.model_ids_file)}
    else:
        raise CivitaiCatalogError("--collection か --model-ids-file のどちらかが必要です")

    items: list[dict[str, Any]] = []
    for index, ref in enumerate(refs, start=1):
        print(f"[{index}/{len(refs)}] model {ref.model_id}", file=sys.stderr)
        model = fetch_model(session, ref.model_id, token=token, base_url=collection_base_url)
        if args.only_lora and model.get("type") not in {"LORA", "LoRA", "LyCORIS"}:
            continue
        if args.all_versions:
            items.extend(normalize_lora_version_entries(model, ref=ref))
        else:
            entry = normalize_lora_entry(model, ref=ref)
            if entry:
                items.append(entry)

    items.sort(key=lambda item: (str(item.get("base_model") or ""), str(item.get("name") or "").lower()))
    return {
        "catalog_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "item_count": len(items),
        "items": items,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Civitai collection から LoRA カタログ YAML/JSONL を生成する")
    parser.add_argument("--collection", help="Civitai collection URL または collection id")
    parser.add_argument("--model-ids-file", help="collection API が使えない場合の model URL/id リスト")
    parser.add_argument("--token", help="Civitai API token。未指定時は CIVITAI_API_TOKEN を参照")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="出力 YAML パス")
    parser.add_argument("--jsonl-output", type=Path, default=DEFAULT_JSONL_OUTPUT, help="出力 JSONL パス")
    parser.add_argument("--no-jsonl", action="store_true", help="JSONL を出力しない")
    parser.add_argument("--only-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--all-versions",
        action="store_true",
        help="各 model の全 modelVersions を version 単位で出力する",
    )
    parser.add_argument("--download-catalog", type=Path, help="指定 YAML catalog の items を download する")
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR, help="download 先ディレクトリ")
    parser.add_argument("--force-download", action="store_true", help="既存ファイルも再downloadする")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.download_catalog:
        try:
            counts = download_catalog(args)
        except CivitaiCatalogError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(counts, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 0

    if args.all_versions:
        if args.output == DEFAULT_OUTPUT:
            args.output = DEFAULT_RAW_OUTPUT
        if args.jsonl_output == DEFAULT_JSONL_OUTPUT:
            args.jsonl_output = DEFAULT_RAW_JSONL_OUTPUT
    try:
        catalog = build_catalog(args)
        write_catalog(catalog, args.output, None if args.no_jsonl else args.jsonl_output)
    except CivitaiCatalogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}", file=sys.stderr)
    if not args.no_jsonl:
        print(f"wrote {args.jsonl_output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
