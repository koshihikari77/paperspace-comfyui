#!/usr/bin/env python3
"""Character LoRA database (Nijisanji / Hololive, Illustrious only).

Subcommands
  roster   Rebuild chara_db/rosters/<franchise>.json from English Wikipedia
           (gender / exclusions come from chara_db/overrides.yaml).
  crawl    Collect Illustrious LoRAs from Civitai into the local cache.
  build    Write chara_db/catalog/<franchise>.csv (all candidates) and
           chara_db/loras.yaml (one selected LoRA per character).
  upload   Download selected LoRAs that are not yet in the HF mirror from
           Civitai (CIVITAI_API_KEY), verify SHA256 and upload them.
  fetch    Download the LoRAs of the given characters from the HF mirror.
  list     Print character ids that have a LoRA in the mirror.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DB = REPO_ROOT / "chara_db"
CACHE = REPO_ROOT / "workspace" / "chara_db_cache"
LORAS_YAML = DB / "loras.yaml"
OVERRIDES_YAML = DB / "overrides.yaml"
FRANCHISES = {
    "nijisanji": {
        "wiki_page": "Nijisanji",
        "wiki_start": "Members (Livers)",
        "wiki_end": "Sub-Units",
        "queries": ["nijisanji", "にじさんじ"],
        "tags": ["nijisanji"],
    },
    "hololive": {
        "wiki_page": "Hololive Production",
        "wiki_start": "== Talents",
        "wiki_end": "== Selected discography",
        "queries": ["hololive", "ホロライブ"],
        "tags": ["hololive"],
    },
}
CIVITAI = "https://civitai.com/api/v1"
UA = "chara-lora-db/1.0"
BASE_MODELS = {"Illustrious"}
# Selection rule: prefer general models over single-outfit ones; among those
# whose downloads are at least POPULAR_RATIO of the most-downloaded one,
# take the newest.
POPULAR_RATIO = 0.5
SINGLE_OUTFIT = re.compile(
    r"outfit|costume|衣装|制服|swim|bikini|水着|birthday|dress|holonatsu|paradise|casual|babydoll|mask|uniform|"
    r"pajama|maid|bunny|kimono|浴衣|ver\.|version|jirai|punk|commission", re.I)
MULTI_OUTFIT = re.compile(r"\d+\s*(outfits|costumes|cos)|various outfits|multiple outfits", re.I)


# ---------------------------------------------------------------- helpers
def norm(text: str | None) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"[\s・･_\-.'’()\[\]/|,:;!?「」【】+]+", " ", text).strip()


def slug(en: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", unicodedata.normalize("NFKC", en).lower()).strip("_")


def load_yaml(path: Path, default):
    if not path.exists():
        return default
    return yaml.safe_load(path.read_text(encoding="utf-8")) or default


def dump_yaml(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")


def http_json(session, url, params=None, tries=5):
    import requests

    for attempt in range(tries):
        try:
            response = session.get(url, params=params, timeout=90)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(3 * (attempt + 1))
    raise SystemExit(f"request failed: {url} {params}")


def civitai_session():
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = UA
    key = os.environ.get("CIVITAI_API_KEY", "").strip()
    if key:
        session.headers["Authorization"] = f"Bearer {key}"
    return session


# ---------------------------------------------------------------- roster
def parse_wiki_roster(wikitext: str) -> list[dict]:
    text = re.sub(r"<ref[^>]*/>", "", wikitext)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"\{\{small\|\{\{bracket\|.*?\}\}\}\}", "", text)
    text = re.sub(r"\{\{efn[^{}]*\}\}", "", text)

    def link(s: str) -> str:
        return re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s)

    sections: list[str] = []
    group = None
    out = []
    for line in text.splitlines():
        heading = re.match(r"^(=+)\s*(.*?)\s*=+\s*$", line)
        if heading:
            level = len(heading.group(1))
            sections = sections[: max(level - 2, 0)]
            sections.append(link(heading.group(2)))
            group = None
            continue
        bold = re.match(r"^'''(.+?)'''", line)
        if bold:
            group = link(bold.group(1))
            continue
        if not line.startswith("*"):
            continue
        item = line.lstrip("* ").strip()
        nihongo = re.search(r"\{\{Nihongo\|([^|]*)\|([^|}]*)", item)
        if nihongo:
            en, ja = link(nihongo.group(1)).strip(), nihongo.group(2).strip()
        else:
            lang = re.search(r"\{\{lang\|[a-z]+\|([^}]*)\}\}", item)
            ja = lang.group(1) if lang else ""
            en = link(re.sub(r"\{\{.*?\}\}", "", item)).strip(" ()")
        if en:
            out.append({"en": en, "ja": ja, "sections": list(sections), "group": group})
    return out


def cmd_roster(args) -> None:
    import requests

    overrides = load_yaml(OVERRIDES_YAML, {})
    session = requests.Session()
    session.headers["User-Agent"] = UA
    for name in args.franchise:
        conf = FRANCHISES[name]
        data = http_json(session, "https://en.wikipedia.org/w/api.php",
                         dict(action="parse", page=conf["wiki_page"], prop="wikitext", format="json"))
        wikitext = data["parse"]["wikitext"]["*"]
        start = wikitext.find(conf["wiki_start"])
        end = wikitext.find(conf["wiki_end"], start)
        rows = parse_wiki_roster(wikitext[start:end])
        ov = overrides.get(name, {})
        chars: dict[str, dict] = {}
        for row in rows:
            cid = slug(row["en"])
            sec = row["sections"][-1]
            former = any("former" in s.lower() for s in row["sections"])
            entry = chars.setdefault(cid, {
                "id": cid, "en": row["en"], "ja": row["ja"], "branch": None,
                "groups": [], "status": "former" if former else "current",
                "gender": None, "aliases": [], "include": True, "exclude_reason": None,
            })
            label = row["group"] if former and row["group"] else sec
            if entry["branch"] is None:
                entry["branch"] = label
            if row["group"] and row["group"] not in entry["groups"]:
                entry["groups"].append(row["group"])
        apply_overrides(name, chars, ov)
        path = DB / "rosters" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(chars.values(), key=lambda c: c["id"]), ensure_ascii=False, indent=1) + "\n")
        included = sum(c["include"] for c in chars.values())
        print(f"{name}: {len(chars)} characters, {included} included -> {path.relative_to(REPO_ROOT)}")


def apply_overrides(name: str, chars: dict[str, dict], ov: dict) -> None:
    excluded_branches = [b.lower() for b in ov.get("exclude_branches", [])]
    male_branches = [b.lower() for b in ov.get("male_branches", [])]
    male = set(ov.get("male", []))
    female = set(ov.get("female", []))
    for cid, c in chars.items():
        branch = (c["branch"] or "").lower()
        if any(b in branch for b in male_branches):
            c["gender"] = "male"
        if cid in male:
            c["gender"] = "male"
        if cid in female:
            c["gender"] = "female"
        if c["gender"] is None and ov.get("default_gender"):
            c["gender"] = ov["default_gender"]
        c["aliases"] = list(ov.get("aliases", {}).get(cid, []))
        if any(b in branch for b in excluded_branches):
            c["include"], c["exclude_reason"] = False, "branch"
        elif c["gender"] == "male":
            c["include"], c["exclude_reason"] = False, "male"
        elif c["gender"] is None:
            c["include"], c["exclude_reason"] = False, "gender unknown"
        if cid in ov.get("exclude", []):
            c["include"], c["exclude_reason"] = False, "manual"


def load_roster(name: str) -> list[dict]:
    path = DB / "rosters" / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run: chara_db.py roster")
    return json.loads(path.read_text())


# ---------------------------------------------------------------- crawl
def crawl_models(session, params: dict, models: dict, label: str) -> int:
    count = 0
    query = dict(params, limit=100, nsfw="true", types="LORA")
    while True:
        data = http_json(session, f"{CIVITAI}/models", query)
        if not data:
            break
        for item in data.get("items", []):
            models.setdefault(item["id"], item).setdefault("_hits", [])
            if label not in models[item["id"]]["_hits"]:
                models[item["id"]]["_hits"].append(label)
            count += 1
        cursor = (data.get("metadata") or {}).get("nextCursor")
        if not cursor:
            break
        query = dict(params, limit=100, nsfw="true", types="LORA", cursor=cursor)
    return count


def cmd_crawl(args) -> None:
    session = civitai_session()
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in args.franchise:
        conf = FRANCHISES[name]
        models: dict[int, dict] = {}
        for base in sorted(BASE_MODELS):
            for tag in conf["tags"]:
                print(name, f"tag:{tag}", crawl_models(session, {"tag": tag, "baseModels": base}, models, f"tag:{tag}"), flush=True)
            for q in conf["queries"]:
                print(name, f"q:{q}", crawl_models(session, {"query": q, "baseModels": base}, models, f"q:{q}"), flush=True)
            if not args.skip_names:
                targets = [c for c in load_roster(name) if c["include"]]
                for i, c in enumerate(targets, 1):
                    for term in {c["en"], *c["aliases"]}:
                        crawl_models(session, {"query": term, "baseModels": base}, models, "q:name")
                    if i % 20 == 0:
                        print(name, f"name queries {i}/{len(targets)} total={len(models)}", flush=True)
        out = CACHE / f"civitai_{name}.json"
        out.write_text(json.dumps(list(models.values()), ensure_ascii=False))
        print(f"{name}: {len(models)} models -> {out}")


# ---------------------------------------------------------------- build
def match_keys(c: dict) -> list:
    keys = []
    for en in [c["en"], *c["aliases"]]:
        n = norm(en)
        parts = n.split()
        if len(parts) >= 2:
            keys += [n, " ".join(parts[::-1]), n.replace(" ", "")]
        elif len(n) >= 5 or not n.isascii():
            keys.append(n)
        elif n:
            keys.append(("word", n))
    ja = norm(c["ja"])
    if len(ja.replace(" ", "")) >= 3:
        keys += [ja, ja.replace(" ", "")]
    return keys


def has_key(key: str, text: str) -> bool:
    if key.isascii():
        return re.search(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])", text) is not None
    return key in text


def model_context(model: dict) -> str:
    versions = model.get("modelVersions") or []
    desc = re.sub(r"<[^>]+>", " ", model.get("description") or "")
    return norm(" ".join([model["name"], " ".join(model.get("tags") or []), desc,
                          " ".join(w for v in versions[:3] for w in (v.get("trainedWords") or []))]))


def match_model(model: dict, roster: list[dict], franchise_words: list[str]) -> list[str]:
    """Return roster ids named by the model. The model must mention the
    franchise (name/tags/description/trigger words); a plain "vtuber"
    mention is accepted only for full two-part names."""
    versions = model.get("modelVersions") or []
    name = norm(model["name"])
    extra = norm(" ".join(model.get("tags") or []) + " " + " ".join(
        w for v in versions[:3] for w in (v.get("trainedWords") or [])))
    context = model_context(model)
    in_franchise = any(norm(w) in context for w in franchise_words)
    vtuber = in_franchise or "vtuber" in context or "virtual youtuber" in context
    if not vtuber:
        # Without any franchise context only distinctive full names count:
        # a Japanese full name, or a long two-part romanized name.
        hits = [c["id"] for c in roster if len(norm(c["en"]).split()) >= 2 and any(
            not isinstance(k, tuple) and ((not k.isascii() and len(k.replace(" ", "")) >= 3)
                                          or (" " in k and len(k) >= 12)) and has_key(k, name)
            for k in c["_keys"])]
        return hits
    for field in (name, extra):
        hits = []
        for c in roster:
            for key in c["_keys"]:
                if isinstance(key, tuple):
                    if field is name and in_franchise and has_key(key[1], name):
                        hits.append(c["id"])
                        break
                elif has_key(key, field) and (in_franchise or " " in key or not key.isascii()):
                    hits.append(c["id"])
                    break
        if hits:
            return hits
    return []


def illustrious_versions(model: dict) -> list[dict]:
    out = []
    for v in model.get("modelVersions") or []:
        if v.get("baseModel") not in BASE_MODELS:
            continue
        files = [f for f in v.get("files") or [] if f.get("type") == "Model" and f["name"].endswith(".safetensors")]
        if not files:
            continue
        f = next((f for f in files if f.get("primary")), files[0])
        out.append({
            "versionId": v["id"], "version": v["name"], "publishedAt": v.get("publishedAt"),
            "file": f["name"], "sizeMB": round((f.get("sizeKB") or 0) / 1024),
            "sha256": ((f.get("hashes") or {}).get("SHA256") or "").lower(),
            "downloads": (v.get("stats") or {}).get("downloadCount") or 0,
            "trainedWords": v.get("trainedWords") or [],
        })
    return out


def cmd_build(args) -> None:
    db = load_yaml(LORAS_YAML, {"version": 1, "characters": {}})
    chars_db = db.setdefault("characters", {})
    overrides = load_yaml(OVERRIDES_YAML, {})
    mirror = mirror_files() if not args.offline else {}
    for name in args.franchise:
        roster = [c for c in load_roster(name) if c["include"]]
        for c in roster:
            c["_keys"] = match_keys(c)
        raw = json.loads((CACHE / f"civitai_{name}.json").read_text())
        words = FRANCHISES[name]["queries"] + FRANCHISES[name]["tags"]
        candidates: dict[str, list[dict]] = {c["id"]: [] for c in roster}
        for model in raw:
            versions = illustrious_versions(model)
            if not versions:
                continue
            hits = match_model(model, roster, words)
            if len(hits) != 1:
                continue
            latest = max(versions, key=lambda v: v["publishedAt"] or "")
            candidates[hits[0]].append({
                "modelId": model["id"], "name": model["name"],
                "creator": (model.get("creator") or {}).get("username"),
                "downloads": (model.get("stats") or {}).get("downloadCount") or 0,
                "latest": latest, "nversions": len(versions),
            })
        rows = []
        pinned = overrides.get(name, {}).get("pin", {})
        legacy = overrides.get(name, {}).get("legacy_files", {})
        for c in roster:
            cands = sorted(candidates[c["id"]], key=lambda m: -m["downloads"])
            for m in cands:
                rows.append([c["id"], c["en"], c["ja"], c["status"], c["branch"], m["modelId"], m["name"], m["creator"],
                             m["downloads"], m["latest"]["version"], m["latest"]["versionId"], (m["latest"]["publishedAt"] or "")[:10],
                             m["latest"]["sizeMB"], f"https://civitai.com/models/{m['modelId']}"])
            key = f"{name}/{c['id']}"
            entry = chars_db.get(key, {})
            pick = select(cands, pinned.get(c["id"]))
            local = existing_file(name, c["id"], pick, mirror, entry)
            if not pick and not local:
                # No Illustrious LoRA on Civitai: fall back to a legacy mirror file.
                kept = [f for f in legacy.get(c["id"], []) if f"loras/{name}/{f}" in mirror]
                if kept:
                    local = {"file": kept[0], "sha256": mirror[f"loras/{name}/{kept[0]}"].get("sha256"),
                             "note": "legacy mirror file (no Civitai Illustrious candidate)"}
                    entry.pop("civitai", None)
                    entry.pop("trigger", None)
            if not pick and not local:
                chars_db.pop(key, None)
                continue
            entry.update({"franchise": name, "en": c["en"], "ja": c["ja"], "status": c["status"], "branch": c["branch"]})
            if pick:
                v = pick["latest"]
                entry["civitai"] = {"modelId": pick["modelId"], "versionId": v["versionId"], "name": pick["name"],
                                    "version": v["version"], "creator": pick["creator"], "downloads": pick["downloads"],
                                    "publishedAt": (v["publishedAt"] or "")[:10]}
                entry["trigger"] = v["trainedWords"]
            entry["file"] = local["file"] if local else pick["latest"]["file"]
            entry["sha256"] = local["sha256"] if local else pick["latest"]["sha256"]
            entry["hf_path"] = f"loras/{name}/{entry['file']}"
            entry["in_mirror"] = entry["hf_path"] in mirror
            if local and local.get("note"):
                entry["note"] = local["note"]
            else:
                entry.pop("note", None)
            alternates = [f for f in legacy.get(c["id"], []) if f != entry["file"] and f"loras/{name}/{f}" in mirror]
            if alternates:
                entry["alternates"] = alternates
            else:
                entry.pop("alternates", None)
            chars_db[key] = entry
        out = DB / "catalog" / f"{name}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "en", "ja", "status", "branch", "modelId", "name", "creator", "downloads",
                        "latestVersion", "latestVersionId", "publishedAt", "sizeMB", "url"])
            w.writerows(rows)
        have = sum(1 for c in roster if candidates[c["id"]])
        print(f"{name}: {len(roster)} target characters, {have} with Illustrious LoRA, {len(rows)} candidates -> {out.relative_to(REPO_ROOT)}")
    db["characters"] = dict(sorted(chars_db.items()))
    db["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    db["rule"] = (f"Illustrious only; general models before single-outfit ones; among candidates with downloads >= "
                  f"{POPULAR_RATIO:.0%} of the top one the newest wins; overrides.yaml pin wins")
    dump_yaml(LORAS_YAML, db)
    missing = [k for k, v in db["characters"].items() if not v.get("in_mirror")]
    print(f"loras.yaml: {len(db['characters'])} characters, {len(missing)} not yet in mirror")


def select(cands: list[dict], pin: int | None) -> dict | None:
    if pin:
        return next((m for m in cands if m["modelId"] == pin), None)
    if not cands:
        return None
    general = [m for m in cands if MULTI_OUTFIT.search(m["name"]) or not SINGLE_OUTFIT.search(m["name"])]
    cands = general or cands
    top = max(m["downloads"] for m in cands)
    popular = [m for m in cands if m["downloads"] >= top * POPULAR_RATIO]
    return max(popular, key=lambda m: m["latest"]["publishedAt"] or "")


def existing_file(name: str, cid: str, pick: dict | None, mirror: dict, entry: dict) -> dict | None:
    """Reuse a file already in the mirror (no renaming) when it is the picked
    version, or when the character has an explicitly kept file."""
    if pick:
        sha = pick["latest"]["sha256"]
        for path, info in mirror.items():
            if path.startswith(f"loras/{name}/") and info.get("sha256") == sha:
                return {"file": path.rsplit("/", 1)[1], "sha256": sha}
    keep = load_yaml(OVERRIDES_YAML, {}).get(name, {}).get("keep_file", {}).get(cid)
    if keep:
        path = f"loras/{name}/{keep}"
        if path in mirror:
            return {"file": keep, "sha256": mirror[path].get("sha256"), "note": "kept existing mirror file"}
    return None


# ---------------------------------------------------------------- HF mirror
def mirror_repo() -> tuple[str, str]:
    conf = load_yaml(REPO_ROOT / "hf-repo.yaml", {})
    return conf["repo"], conf.get("revision", "main")


def mirror_files() -> dict[str, dict]:
    from huggingface_hub import HfApi

    repo, revision = mirror_repo()
    out = {}
    for item in HfApi().list_repo_tree(repo, revision=revision, path_in_repo="loras", recursive=True, expand=True):
        if not hasattr(item, "size"):
            continue
        sha = item.lfs.sha256 if getattr(item, "lfs", None) else None
        out[item.path] = {"sha256": sha, "size": item.size}
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_upload(args) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    if not os.environ.get("CIVITAI_API_KEY"):
        raise SystemExit("CIVITAI_API_KEY is required for upload")
    db = load_yaml(LORAS_YAML, {"characters": {}})
    repo, revision = mirror_repo()
    api = HfApi()
    session = civitai_session()
    todo = [(k, v) for k, v in db["characters"].items() if not v.get("in_mirror") and v.get("civitai")]
    if args.chara:
        todo = [(k, v) for k, v in todo if k in resolve_ids(db, args.chara)]
    if args.limit:
        todo = todo[: args.limit]
    print(f"upload: {len(todo)} files, {args.batch} per commit", flush=True)
    work = Path(tempfile.mkdtemp(prefix="chara-upload-", dir=args.tmp))
    failed = []
    done = 0
    try:
        for b in range(0, len(todo), args.batch):
            batch, ops = todo[b: b + args.batch], []
            for key, entry in batch:
                vid = entry["civitai"]["versionId"]
                dest = work / f"{vid}.safetensors"
                try:
                    with session.get(f"https://civitai.com/api/download/models/{vid}", stream=True, timeout=120) as r:
                        r.raise_for_status()
                        with dest.open("wb") as fh:
                            for chunk in r.iter_content(1 << 22):
                                fh.write(chunk)
                    actual = sha256_file(dest)
                    if entry.get("sha256") and actual != entry["sha256"]:
                        raise RuntimeError(f"sha256 mismatch {actual} != {entry['sha256']}")
                    ops.append((key, entry, dest, actual))
                except Exception as exc:  # keep going; report at the end
                    failed.append((key, str(exc)[:200]))
                    print(f"FAIL download {key}: {exc}", flush=True)
                    dest.unlink(missing_ok=True)
            if ops:
                api.create_commit(
                    repo_id=repo, revision=revision,
                    operations=[CommitOperationAdd(path_in_repo=e["hf_path"], path_or_fileobj=str(d)) for _, e, d, _ in ops],
                    commit_message=f"Add {len(ops)} character LoRAs ({', '.join(k for k, *_ in ops)})"[:400],
                )
                for key, entry, dest, actual in ops:
                    entry["sha256"], entry["in_mirror"] = actual, True
                    done += 1
                    print(f"[{done}/{len(todo)}] OK {key} {entry['file']} ({dest.stat().st_size / 1e6:.0f} MB)", flush=True)
                    dest.unlink(missing_ok=True)
                dump_yaml(LORAS_YAML, db)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"uploaded {done}/{len(todo)}")
    if failed:
        print("failed:", *[f"  {k}: {e}" for k, e in failed], sep="\n")


# ---------------------------------------------------------------- fetch
def resolve_ids(db: dict, wanted: list[str]) -> list[str]:
    chars = db["characters"]
    out = []
    for w in wanted:
        w_norm = norm(w)
        if w in chars:
            out.append(w)
            continue
        hits = [k for k, v in chars.items()
                if k.split("/", 1)[1] == slug(w) or norm(v.get("ja")) == w_norm or norm(v.get("en")) == w_norm]
        if w.endswith("/*"):
            hits = [k for k in chars if k.startswith(w[:-1])]
        if not hits:
            raise SystemExit(f"unknown character: {w} (see chara_db/loras.yaml)")
        out += hits
    return list(dict.fromkeys(out))


def cmd_fetch(args) -> None:
    db = load_yaml(LORAS_YAML, {"characters": {}})
    keys = resolve_ids(db, args.chara)
    entries = [(k, db["characters"][k]) for k in keys]
    paths = [v["hf_path"] for k, v in entries if v.get("in_mirror")]
    fallback = [k for k, v in entries if not v.get("in_mirror") and v.get("alternates")]
    missing = [k for k, v in entries if not v.get("in_mirror") and not v.get("alternates")]
    if fallback:
        print("selected file not in mirror yet, using existing alternates:", ", ".join(fallback))
    if missing:
        print("not in mirror yet (skipped):", ", ".join(missing))
    paths += [f"loras/{v['franchise']}/{f}" for k, v in entries
              for f in v.get("alternates", []) if args.with_alternates or k in fallback]
    root = Path(args.model_root)
    todo = [p for p in paths if not (root / p).exists()]
    print(f"CHARA_LORAS: {len(paths)} selected, {len(paths) - len(todo)} already present, {len(todo)} to download")
    if todo:
        repo, revision = mirror_repo()
        cmd = ["hf", "download", repo, "--revision", revision, "--local-dir", str(root), *todo]
        subprocess.run(cmd, check=True)
    for k, v in entries:
        if v.get("in_mirror"):
            print(f"  {k}: {v['hf_path']}  trigger={', '.join(v.get('trigger') or [])[:80]}")
        elif k in fallback:
            print(f"  {k}: {', '.join(v['alternates'])}  (existing files; trigger words differ from loras.yaml)")


def cmd_list(args) -> None:
    db = load_yaml(LORAS_YAML, {"characters": {}})
    for k, v in db["characters"].items():
        if args.franchise and v["franchise"] not in args.franchise:
            continue
        mark = "" if v.get("in_mirror") else "  (not in mirror)"
        print(f"{k}\t{v.get('ja') or ''}\t{v['file']}{mark}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ("roster", "crawl", "build"):
        p = sub.add_parser(cmd)
        p.add_argument("--franchise", nargs="+", default=list(FRANCHISES), choices=list(FRANCHISES))
        if cmd == "crawl":
            p.add_argument("--skip-names", action="store_true", help="only tag/keyword search, no per-character queries")
        if cmd == "build":
            p.add_argument("--offline", action="store_true", help="do not list the HF mirror")
    p = sub.add_parser("upload")
    p.add_argument("--chara", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch", type=int, default=10, help="files per HF commit")
    p.add_argument("--tmp", default=None)
    p = sub.add_parser("fetch")
    p.add_argument("--chara", nargs="+", required=True, help="ids like nijisanji/shiina_yuika, names (椎名唯華) or nijisanji/*")
    p.add_argument("--model-root", default="/app/models")
    p.add_argument("--with-alternates", action="store_true", help="also fetch the legacy mirror files of these characters")
    p = sub.add_parser("list")
    p.add_argument("--franchise", nargs="*")
    args = parser.parse_args()
    {"roster": cmd_roster, "crawl": cmd_crawl, "build": cmd_build, "upload": cmd_upload,
     "fetch": cmd_fetch, "list": cmd_list}[args.cmd](args)


if __name__ == "__main__":
    main()
