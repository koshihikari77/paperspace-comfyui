#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
import sys
import urllib.request
from email.message import Message
from http.client import IncompleteRead
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.error import HTTPError, URLError

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download files described by a YAML catalog into a local directory."
    )
    parser.add_argument("--yaml", required=True, help="Path to the YAML catalog.")
    parser.add_argument("--output-dir", required=True, help="Destination directory.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without fetching.")
    parser.add_argument("--limit", type=int, help="Download at most this many matched items.")
    parser.add_argument(
        "--key",
        action="append",
        default=[],
        help="Only include items whose key exactly matches. Repeatable.",
    )
    parser.add_argument(
        "--key-regex",
        help="Only include items whose key matches this regular expression.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Only include items containing this tag. Repeatable.",
    )
    parser.add_argument(
        "--match-all-tags",
        action="store_true",
        help="Require all --tag values instead of any one of them.",
    )
    parser.add_argument(
        "--nsfw",
        choices=["any", "true", "false"],
        default="any",
        help="Filter by the item's nsfw field.",
    )
    parser.add_argument(
        "--filename-field",
        default="file_name",
        help="Item field used as the destination filename. Defaults to file_name.",
    )
    parser.add_argument(
        "--url-field",
        default="download_url",
        help="Item field used as the source URL. Defaults to download_url.",
    )
    parser.add_argument(
        "--token-env",
        help="Environment variable name containing a bearer token to send.",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Additional HTTP header in 'Name: Value' form. Repeatable.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds. Defaults to 120.",
    )
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Verify hashes.SHA256 after download when present.",
    )
    parser.add_argument(
        "--rewrite-civitai-host-from-yaml",
        action="store_true",
        help="Rewrite civitai download URLs to the YAML source.base_url host when present.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for transient HTTP failures. Defaults to 3.",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=2.0,
        help="Initial wait between retries in seconds. Defaults to 2.0.",
    )
    parser.add_argument(
        "--user-agent",
        default="Mozilla/5.0 (X11; Linux x86_64) PythonYamlDownloader/1.0",
        help="User-Agent header to send on requests.",
    )
    parser.add_argument(
        "--trust-content-disposition",
        action="store_true",
        help="Use the response filename from Content-Disposition when present.",
    )
    return parser.parse_args()


def load_catalog(yaml_path: Path) -> tuple[dict | None, list[dict]]:
    with yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if isinstance(data, dict):
        items = data.get("items")
        if items is None:
            raise SystemExit("YAML does not contain an 'items' list")
        catalog = data
    elif isinstance(data, list):
        items = data
        catalog = None
    else:
        raise SystemExit("YAML root must be a mapping or a list")

    if not isinstance(items, list):
        raise SystemExit("'items' must be a list")
    return catalog, [item for item in items if isinstance(item, dict)]


def compile_headers(raw_headers: list[str], token_env: str | None) -> dict[str, str]:
    headers: dict[str, str] = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) PythonYamlDownloader/1.0",
        "Accept": "*/*",
    }

    for raw in raw_headers:
        if ":" not in raw:
            raise SystemExit(f"Invalid header format: {raw!r}")
        name, value = raw.split(":", 1)
        headers[name.strip()] = value.strip()

    if token_env:
        token = os.environ.get(token_env, "").strip()
        if not token:
            raise SystemExit(f"Environment variable is empty or missing: {token_env}")
        headers["Authorization"] = f"Bearer {token}"

    return headers


def matches_tags(item_tags: list[str], wanted_tags: list[str], match_all: bool) -> bool:
    if not wanted_tags:
        return True
    tag_set = set(item_tags)
    wanted = set(wanted_tags)
    return wanted.issubset(tag_set) if match_all else bool(tag_set & wanted)


def matches_nsfw(item: dict, nsfw_filter: str) -> bool:
    if nsfw_filter == "any":
        return True
    if nsfw_filter == "true":
        return bool(item.get("nsfw")) is True
    return bool(item.get("nsfw")) is False


def select_items(items: list[dict], args: argparse.Namespace) -> list[dict]:
    key_regex = re.compile(args.key_regex) if args.key_regex else None
    selected: list[dict] = []

    for item in items:
        key = str(item.get("key", ""))
        tags = item.get("tags") or []
        if not isinstance(tags, list):
            tags = []

        if args.key and key not in args.key:
            continue
        if key_regex and not key_regex.search(key):
            continue
        if not matches_tags([str(tag) for tag in tags], args.tag, args.match_all_tags):
            continue
        if not matches_nsfw(item, args.nsfw):
            continue

        selected.append(item)
        if args.limit is not None and len(selected) >= args.limit:
            break

    return selected


def destination_name(item: dict, filename_field: str) -> str:
    value = item.get(filename_field) or item.get("file_name") or item.get("key")
    if not value:
        raise SystemExit(f"Item is missing both {filename_field!r}, 'file_name', and 'key'")
    return Path(str(value)).name


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def with_query_param(url: str, name: str, value: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query[name] = value
    return urlunparse(parsed._replace(query=urlencode(query)))


def rewrite_civitai_host(url: str, base_url: str | None) -> str:
    if not base_url:
        return url

    parsed = urlparse(url)
    base = urlparse(base_url)
    if parsed.netloc not in {"civitai.com", "www.civitai.com", "civitai.red", "www.civitai.red"}:
        return url
    if not base.scheme or not base.netloc:
        return url
    return urlunparse(parsed._replace(scheme=base.scheme, netloc=base.netloc))


def build_download_url(
    item: dict,
    url_field: str,
    token_env: str | None,
    rewrite_host: bool,
    catalog_base_url: str | None,
) -> str:
    url = item.get(url_field)
    if not url:
        raise SystemExit(f"Item is missing URL field {url_field!r}: {item.get('key', '')}")

    final_url = str(url)
    if rewrite_host:
        final_url = rewrite_civitai_host(final_url, catalog_base_url)

    if token_env:
        token = os.environ.get(token_env, "").strip()
        if token and urlparse(final_url).netloc in {
            "civitai.com",
            "www.civitai.com",
            "civitai.red",
            "www.civitai.red",
        }:
            final_url = with_query_param(final_url, "token", token)

    return final_url


def download_file(url: str, destination: Path, headers: dict[str, str], timeout: int) -> None:
    request = urllib.request.Request(url, headers=headers)
    temp_path = destination.with_name(f"{destination.name}.part")
    with urllib.request.urlopen(request, timeout=timeout) as response, temp_path.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    temp_path.replace(destination)


def filename_from_headers(headers: Message) -> str | None:
    content_disposition = headers.get("Content-Disposition")
    if not content_disposition:
        return None

    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', content_disposition, re.IGNORECASE)
    if not match:
        return None

    return Path(match.group(1).strip()).name or None


def retry_after_seconds(error: HTTPError) -> float | None:
    value = error.headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def is_retryable(error: Exception) -> bool:
    if isinstance(error, HTTPError):
        return error.code in {408, 409, 425, 429, 500, 502, 503, 504}
    return isinstance(error, (URLError, TimeoutError, IncompleteRead))


def download_with_retries(
    url: str,
    destination: Path,
    headers: dict[str, str],
    timeout: int,
    retries: int,
    retry_wait: float,
    trust_content_disposition: bool,
) -> Path:
    attempt = 0
    current_destination = destination

    while True:
        try:
            request = urllib.request.Request(url, headers=headers)
            temp_path = current_destination.with_name(f"{current_destination.name}.part")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if trust_content_disposition:
                    header_name = filename_from_headers(response.headers)
                    if header_name and header_name != current_destination.name:
                        current_destination = current_destination.with_name(header_name)
                        temp_path = current_destination.with_name(f"{current_destination.name}.part")
                with temp_path.open("wb") as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
            temp_path.replace(current_destination)
            return current_destination
        except Exception as exc:
            temp_path = current_destination.with_name(f"{current_destination.name}.part")
            temp_path.unlink(missing_ok=True)
            if attempt >= retries or not is_retryable(exc):
                raise
            wait = retry_after_seconds(exc) if isinstance(exc, HTTPError) else None
            if wait is None:
                wait = retry_wait * (2 ** attempt)
            print(
                f"RETRY {attempt + 1}/{retries}: {current_destination.name} after {wait:.1f}s due to {type(exc).__name__}",
                file=sys.stderr,
            )
            time.sleep(wait)
            attempt += 1


def main() -> int:
    args = parse_args()
    yaml_path = Path(args.yaml)
    output_dir = Path(args.output_dir)

    catalog, items = load_catalog(yaml_path)
    headers = compile_headers(args.header, args.token_env)
    headers["User-Agent"] = args.user_agent
    selected = select_items(items, args)
    catalog_base_url = None
    if isinstance(catalog, dict):
        source = catalog.get("source")
        if isinstance(source, dict):
            base_url = source.get("base_url")
            if isinstance(base_url, str):
                catalog_base_url = base_url

    if not selected:
        print("No matching items found.", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"YAML: {yaml_path}")
    print(f"Matched items: {len(selected)}")
    print(f"Output dir: {output_dir}")

    downloaded = 0
    skipped = 0

    for item in selected:
        key = str(item.get("key", ""))
        url = build_download_url(
            item=item,
            url_field=args.url_field,
            token_env=args.token_env,
            rewrite_host=args.rewrite_civitai_host_from_yaml,
            catalog_base_url=catalog_base_url,
        )

        filename = destination_name(item, args.filename_field)
        destination = output_dir / filename

        if destination.exists() and not args.force:
            print(f"SKIP exists: {destination.name} ({key})")
            skipped += 1
            continue

        print(f"{'PLAN' if args.dry_run else 'GET '} {destination.name} <- {url}")
        if args.dry_run:
            continue

        destination = download_with_retries(
            url=str(url),
            destination=destination,
            headers=headers,
            timeout=args.timeout,
            retries=args.retries,
            retry_wait=args.retry_wait,
            trust_content_disposition=args.trust_content_disposition,
        )

        if args.verify_sha256:
            expected = (((item.get("hashes") or {}) if isinstance(item.get("hashes"), dict) else {}).get("SHA256"))
            if expected:
                actual = compute_sha256(destination)
                if actual != str(expected).upper():
                    destination.unlink(missing_ok=True)
                    raise SystemExit(
                        f"SHA256 mismatch for {destination.name}: expected {expected}, got {actual}"
                    )

        downloaded += 1

    print(f"Downloaded: {downloaded}")
    print(f"Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
