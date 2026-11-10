"""Gate 2-3: load Release 17 TeleQnA questions and chunk the configured TSpec-LLM specifications.

Reads TSpec-LLM Markdown from a directory or zip. TeleQnA candidates must carry the
configured Release tag; accepted questions can only cite passages from configured specs.
Outputs:
  work/questions_all.jsonl, work/passages.jsonl, work/corpus_manifest.json
Usage: python 01_prepare_data.py
"""
from __future__ import annotations

import datetime as dt
import glob
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from common import load_config, ntok, sha256_file, split_sentences, tokenizer, work, write_json, write_jsonl

HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
CLAUSE = re.compile(r"^([A-Z]?\d+[a-z]?(?:\.\d+[a-zA-Z]?)*)\s+(.+)$")
ANNEX = re.compile(r"^Annex\s+([A-Z])\b(.*)$")
TABLE_BORDER = re.compile(r"^\s*\+[-=+:]+\+\s*$")
VERSION = re.compile(r"3GPP\s+TS\s+(\d+\.\d+)\s+V(\d+\.\d+\.\d+)[\s\S]{0,50}?\((\d{4}-\d{2})\)")
NORMATIVE = re.compile(r"\b(shall not|shall|should|may)\b", re.I)
RELEASE_TAG = re.compile(r"\[3GPP Release (\d+)\]")


def one_zip(cfg: dict, pattern: str) -> Path:
    hits = sorted(glob.glob(str(Path(cfg["paths"]["raw_dir"]) / pattern)))
    if not hits:
        raise SystemExit(f"No zip matching {pattern} in {cfg['paths']['raw_dir']}")
    return Path(hits[-1])


def load_spec_documents(cfg: dict, wanted: set[str]) -> tuple[list[dict], dict]:
    source = cfg["paths"].get("tspec_source", "zip")
    release = cfg["corpus"]["release"]
    documents = []
    if source == "directory":
        root = Path(cfg["paths"]["tspec_dir"])
        release_root = root / f"Rel-{release}"
        if not release_root.is_dir():
            raise SystemExit(f"TSpec release folder not found: {release_root}")
        matches: dict[str, list[Path]] = {spec: [] for spec in wanted}
        for path in release_root.rglob("*.md"):
            spec = path.name.split("-")[0]
            if spec in matches:
                matches[spec].append(path)
        missing = sorted(spec for spec, paths in matches.items() if not paths)
        duplicates = {spec: [str(p) for p in paths] for spec, paths in matches.items() if len(paths) > 1}
        if missing or duplicates:
            raise SystemExit(f"Incomplete or ambiguous TSpec Release {release} checkout; missing={missing}, duplicates={duplicates}")
        for spec in sorted(matches):
            path = matches[spec][0]
            raw = path.read_bytes()
            if raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise SystemExit(f"Git LFS content is not downloaded yet: {path}")
            documents.append({"name": str(path.relative_to(root)), "text": raw.decode("utf-8", errors="replace"),
                              "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        return documents, {"kind": "directory", "root": str(root), "release": release,
                           "files": [{"file": d["name"], "bytes": d["bytes"], "sha256": d["sha256"]}
                                     for d in documents]}

    if source != "zip":
        raise SystemExit(f"Unsupported paths.tspec_source: {source}")
    zpath = one_zip(cfg, cfg["paths"]["tspec_zip_glob"])
    with zipfile.ZipFile(zpath) as archive:
        members = [name for name in archive.namelist() if name.endswith(".md") and "/.cache/" not in name
                   and f"/Rel-{release}/" in name and Path(name).name.split("-")[0] in wanted]
        for name in sorted(members):
            raw = archive.read(name)
            documents.append({"name": name, "text": raw.decode("utf-8", errors="replace"),
                              "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    missing = wanted - {Path(d["name"]).name.split("-")[0] for d in documents}
    if missing:
        raise SystemExit(f"Missing configured specs in TSpec zip: {sorted(missing)}")
    return documents, {"kind": "zip", "file": zpath.name, "sha256": sha256_file(zpath),
                       "files": [{"file": d["name"], "bytes": d["bytes"], "sha256": d["sha256"]}
                                 for d in documents]}


# ---------------------------------------------------------------- TeleQnA
def load_questions(cfg: dict, zpath: Path) -> list[dict]:
    with zipfile.ZipFile(zpath) as z:
        member = next(n for n in z.namelist() if n.endswith("TeleQnA.json"))
        data = json.loads(z.read(member))
    rows = []
    target_release = int(cfg["corpus"]["release"])
    for qid, q in data.items():
        if q["category"] not in cfg["benchmark"]["categories"]:
            continue
        opts = sorted(((int(k.split()[1]), v) for k, v in q.items() if re.fullmatch(r"option \d+", k)))
        m = re.match(r"option\s*(\d+)", q["answer"])
        tag = RELEASE_TAG.search(q["question"])
        release_tag = int(tag.group(1)) if tag else None
        if release_tag != target_release:
            continue
        rows.append({
            "question_id": qid, "category": q["category"], "question": q["question"],
            "options": [t for _, t in opts], "answer": int(m.group(1)),
            "explanation": q.get("explanation", ""), "release_tag": release_tag,
        })
    return rows


# ---------------------------------------------------------------- TSpec chunking
def clean_lines(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        if TABLE_BORDER.match(ln):
            continue
        if ln.lstrip().startswith("|"):
            ln = re.sub(r"\s{2,}", " ", ln)
        out.append(ln.rstrip())
    return out


def sections(md_text: str):
    """Yield (clause, title, body_text) per heading; text before the first heading (cover page) is dropped."""
    cur = None
    buf: list[str] = []
    for line in md_text.splitlines():
        h = HEADING.match(line)
        if h:
            if cur:
                yield (*cur, "\n".join(clean_lines(buf)).strip())
            title = h.group(2)
            m, a = CLAUSE.match(title), ANNEX.match(title)
            cur = (m.group(1), m.group(2)) if m else ((f"Annex {a.group(1)}", a.group(2).strip(" :")) if a else (title, title))
            buf = []
        elif cur:
            buf.append(line)
    if cur:
        yield (*cur, "\n".join(clean_lines(buf)).strip())


def hard_split(cfg: dict, text: str, hi: int) -> list[str]:
    ids = tokenizer(cfg).encode(text, add_special_tokens=False)
    return [tokenizer(cfg).decode(ids[i:i + hi]) for i in range(0, len(ids), hi)]


def units(cfg: dict, para: str, hi: int) -> list[str]:
    """Split an over-long paragraph into packable units (table rows or sentences)."""
    parts = para.splitlines() if para.lstrip().startswith("|") else split_sentences(para)
    out = []
    for p in parts:
        out.extend(hard_split(cfg, p, hi) if ntok(cfg, p) > hi else [p])
    return out


def pack(cfg: dict, pieces: list[str], hi: int, sep: str) -> list[str]:
    chunks, cur, cur_t = [], [], 0
    for piece in pieces:
        t = ntok(cfg, piece)
        if cur and cur_t + t > hi:
            chunks.append(sep.join(cur))
            cur, cur_t = [], 0
        cur.append(piece)
        cur_t += t
    if cur:
        chunks.append(sep.join(cur))
    return chunks


def chunk_section(cfg: dict, heading_line: str, body: str, lo: int, hi: int) -> list[str]:
    budget = max(64, hi - ntok(cfg, heading_line) - 2)
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    for p in paras:
        pieces.extend(units(cfg, p, budget) if ntok(cfg, p) > budget else [p])
    chunks = pack(cfg, pieces, budget, "\n\n") or [""]
    if len(chunks) > 1 and ntok(cfg, chunks[-1]) < lo and ntok(cfg, chunks[-2] + chunks[-1]) <= hi:
        chunks[-2:] = [chunks[-2] + "\n\n" + chunks[-1]]
    return [f"{heading_line}\n{c}".strip() for c in chunks]


def parent(clause: str) -> str:
    return clause.split(".")[0]


def chunk_spec(cfg: dict, name: str, md_text: str) -> tuple[list[dict], dict]:
    m = VERSION.search(md_text[:5000])
    if not m:
        raise SystemExit(f"Cannot read spec/version header in {name}")
    spec, version, date = m.group(1), "V" + m.group(2), m.group(3)
    lo, hi = cfg["corpus"]["min_tokens"], cfg["corpus"]["max_tokens"]
    skip = {s.lower() for s in cfg["corpus"]["skip_sections"]}

    raw: list[dict] = []
    norm_src = 0
    for clause, title, body in sections(md_text):
        if title.lower() in skip or clause.lower() in skip or not body:
            continue
        norm_src += len(NORMATIVE.findall(body))
        for k, text in enumerate(chunk_section(cfg, f"{clause} {title}", body, lo, hi)):
            raw.append({"clause": clause, "title": title, "part": k, "text": text, "clauses": [clause]})

    # merge consecutive small sections that share a top-level parent (keeps clause list for provenance)
    merged: list[dict] = []
    for c in raw:
        prev = merged[-1] if merged else None
        if (prev and ntok(cfg, prev["text"]) < lo and parent(prev["clause"]) == parent(c["clause"])
                and ntok(cfg, prev["text"] + c["text"]) + 2 <= hi):
            prev["text"] += "\n\n" + c["text"]
            prev["clauses"].append(c["clause"])
        else:
            merged.append(dict(c))

    passages = []
    for c in merged:
        pid = f"{spec.replace('.', '')}-{version}-{c['clause'].replace(' ', '')}-p{c['part']}"
        passages.append({
            "id": pid, "spec": spec, "version": version, "release": cfg["corpus"]["release"],
            "clause": c["clause"], "clauses": c["clauses"], "title": c["title"],
            "header": f"TS {spec} {version} clause {c['clause']} ({c['title']})",
            "source": name, "text": c["text"], "n_tokens": ntok(cfg, c["text"]),
        })
    ids = Counter(p["id"] for p in passages)
    for p in passages:  # disambiguate rare duplicate clause numbers
        if ids[p["id"]] > 1:
            p["id"] += f"-{stable_suffix(p['text'])}"
    norm_out = sum(len(NORMATIVE.findall(p["text"])) for p in passages)
    meta = {"spec": spec, "version": version, "date": date, "file": name, "passages": len(passages),
            "normative_terms_source": norm_src, "normative_terms_passages": norm_out}
    return passages, meta


def stable_suffix(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:6]


def main() -> None:
    cfg = load_config()
    qzip = one_zip(cfg, cfg["paths"]["teleqna_zip_glob"])

    questions = load_questions(cfg, qzip)
    write_jsonl(work(cfg, "questions_all.jsonl"), questions)
    print(f"Release {cfg['corpus']['release']} TeleQnA candidates: {len(questions)} "
          f"{dict(Counter(q['category'] for q in questions))}")

    wanted = {s.replace(".", "") for s in cfg["corpus"]["specs"]}
    passages, specs = [], []
    documents, source_manifest = load_spec_documents(cfg, wanted)
    for doc in tqdm(documents, desc="chunking specs"):
        ps, meta = chunk_spec(cfg, doc["name"], doc["text"])
        meta["source_bytes"] = doc["bytes"]
        meta["source_sha256"] = doc["sha256"]
        passages.extend(ps)
        specs.append(meta)
    missing = wanted - {m["spec"].replace(".", "") for m in specs}
    write_jsonl(work(cfg, "passages.jsonl"), passages)

    toks = sorted(p["n_tokens"] for p in passages)
    norm_ok = all(abs(m["normative_terms_source"] - m["normative_terms_passages"]) <= 0.001 * max(1, m["normative_terms_source"])
                  for m in specs)
    quality = {
        "passages": len(passages), "empty": sum(1 for p in passages if not p["text"].strip()),
        "tokens_p5_p50_p95": [toks[int(len(toks) * q)] for q in (0.05, 0.5, 0.95)] if toks else None,
        "over_max": sum(1 for t in toks if t > cfg["corpus"]["max_tokens"]),
        "unique_ids": len({p["id"] for p in passages}) == len(passages),
        "normative_terms_preserved": norm_ok, "missing_specs": sorted(missing),
    }
    quality["corpus_ok"] = (not missing and quality["empty"] == 0 and quality["unique_ids"] and norm_ok)
    write_json(work(cfg, "corpus_manifest.json"), {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "teleqna_zip": {"file": qzip.name, "sha256": sha256_file(qzip),
                        "downloaded": dt.datetime.fromtimestamp(qzip.stat().st_mtime).isoformat(timespec="seconds"),
                        "release_filter": cfg["corpus"]["release"], "questions_retained": len(questions)},
        "tspec_source": source_manifest,
        "specs": specs, "quality": quality, "tokenizer": cfg["models"]["generator_hf_tokenizer"],
    })
    for m in specs:
        print(f"TS {m['spec']} {m['version']} ({m['date']}): {m['passages']} passages")
    print(f"quality: {quality}")
    if not quality["corpus_ok"]:
        raise SystemExit("Corpus quality flag FAILED - inspect work/corpus_manifest.json before continuing.")


if __name__ == "__main__":
    main()
