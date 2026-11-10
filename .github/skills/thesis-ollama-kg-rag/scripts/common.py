"""Shared helpers: config, I/O, tokenizer budgets and the Ollama client used by every stage."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPTS_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPTS_DIR.parent


# ---------------------------------------------------------------- config / io
def load_config() -> dict:
    env = os.environ.get("THESIS_CONFIG")
    local = Path("thesis_config.yaml")
    path = Path(env) if env else (local if local.exists() else SKILL_DIR / "assets" / "config.yaml")
    cfg = yaml.safe_load(path.read_text())
    cfg["_config_path"] = str(path.resolve())
    cfg["_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    set_seed(cfg["seed"])
    return cfg


def work(cfg: dict, *parts: str) -> Path:
    p = Path(cfg["paths"]["work_dir"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    with Path(path).open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path, rows) -> None:
    with Path(path).open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_int(*parts) -> int:
    """Deterministic per-item seed (Python's hash() is salted per process)."""
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class RawEvidence:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cache: dict[str, tuple[dict, str, str]] = {}

    def check(self, passage: dict, quote: str) -> dict:
        if passage.get("release") != self.cfg["corpus"]["release"] or passage.get("spec") not in self.cfg["corpus"]["specs"]:
            raise ValueError("Evidence is outside the configured release/specification scope")
        source = passage["source"]
        if source not in self.cache:
            root = Path(self.cfg["paths"]["tspec_dir"])
            relative = Path(source)
            expected = Path(f"Rel-{self.cfg['corpus']['release']}") / f"{passage['spec'].split('.')[0]}_series"
            if relative.parent != expected or relative.name.split("-")[0] != passage["spec"].replace(".", ""):
                raise ValueError(f"Unexpected raw source path: {source}")
            md_path = root / relative
            docx_path = md_path.with_suffix(".docx")
            md_raw, docx_raw = md_path.read_bytes(), docx_path.read_bytes()
            if any(raw.startswith(b"version https://git-lfs.github.com/spec/v1") for raw in (md_raw, docx_raw)):
                raise ValueError(f"Raw source download incomplete: {source}")
            with zipfile.ZipFile(docx_path) as archive:
                document = ET.fromstring(archive.read("word/document.xml"))
            namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            docx_text = "\n".join("".join(node.text or "" for node in paragraph.findall(".//w:t", namespace))
                                  for paragraph in document.findall(".//w:p", namespace))
            citation = {"release": passage["release"], "spec": passage["spec"], "version": passage["version"],
                        "folder": str(md_path.parent), "md_name": md_path.name, "md_path": str(md_path),
                        "docx_name": docx_path.name, "docx_path": str(docx_path),
                        "md_sha256": hashlib.sha256(md_raw).hexdigest(),
                        "docx_sha256": hashlib.sha256(docx_raw).hexdigest()}
            self.cache[source] = citation, md_raw.decode("utf-8"), docx_text
        citation, md_text, docx_text = self.cache[source]
        md_ok = quote_in_text(quote, md_text)
        docx_ok = quote_in_text(quote, docx_text)
        return citation | {"passage_id": passage["id"], "clause": passage["clause"], "quote": quote,
                           "raw_md_ok": md_ok, "raw_docx_ok": docx_ok,
                           "source_validated": md_ok and docx_ok}


def load_split(cfg: dict, split: str) -> list[str]:
    return [line.strip() for line in work(cfg, "split", f"{split}_ids.txt").read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------- tokens
@lru_cache(maxsize=4)
def _tokenizer(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def tokenizer(cfg: dict):
    return _tokenizer(cfg["models"]["generator_hf_tokenizer"])


@lru_cache(maxsize=512)
def _token_count(name: str, text: str) -> int:
    return len(_tokenizer(name).encode(text, add_special_tokens=False))


def ntok(cfg: dict, text: str) -> int:
    if not text:
        return 0
    if len(text) > 4096:
        return len(tokenizer(cfg).encode(text, add_special_tokens=False))
    return _token_count(cfg["models"]["generator_hf_tokenizer"], text)


_SENT = re.compile(r"(?<=[.;!?])\s+(?=[A-Z0-9(\-\"'\[])|\n+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]


def trim_to_tokens(cfg: dict, text: str, budget: int) -> str:
    """Longest sentence-boundary prefix of `text` within `budget` tokens ('' if none fits)."""
    if budget <= 0:
        return ""
    if ntok(cfg, text) <= budget:
        return text
    kept: list[str] = []
    for s in split_sentences(text):
        if ntok(cfg, " ".join(kept + [s])) > budget:
            break
        kept.append(s)
    return " ".join(kept)


# ---------------------------------------------------------------- ollama
_CLIENT = None


def ollama_client(cfg: dict):
    global _CLIENT
    if _CLIENT is None:
        import ollama
        _CLIENT = ollama.Client(host=cfg["ollama"]["host"], timeout=cfg["ollama"]["timeout_s"])
    return _CLIENT


def parse_json(text: str):
    """Parse JSON with multiple fallback strategies for malformed LLM output.
    
    Returns:
        A dict if valid JSON object found, None otherwise.
        Never returns a string, even if a string is valid JSON.
    """
    if not text:
        return None
        
    try:
        result = json.loads(text)
        # Only return dicts, not strings or other JSON types
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Attempt to extract outermost { ... }
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    
    # Handle trailing unclosed JSON caused by token budget cuts
    s = (text or "").strip()
    if s.startswith("{"):
        # Close trailing incomplete object or array
        for suffix in ("}", "]}", '"]}', '"}]}', '}]}', '"}]}}', '"}}', '}}}'):
            try:
                result = json.loads(s + suffix)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
    
    # Try to recover from Ollama schema format mode issues
    # Sometimes Ollama produces lines like: \n  "key": "value"
    # Try to build a minimal object
    if '"' in text and ':' in text:
        # Extract all key-value pairs
        pairs = re.findall(r'"([^"]+)"\s*:\s*([^,}]*?)(?=[,}]|\n|$)', text)
        if pairs:
            try:
                obj = {}
                for key, val in pairs:
                    val = val.strip().strip('"\'')
                    obj[key] = val
                if obj:  # Only return if we found at least one pair
                    return obj
            except:
                pass
    
    return None


def llm(cfg: dict, role: str, messages: list[dict], schema: dict | None = None,
        seed: int | None = None, max_tokens: int | None = None) -> dict:
    """One chat call with temperature 0, fixed seed, explicit num_ctx and retries.

    Raises after `max_retries` so callers can log an execution error (never a silent exclusion).
    """
    o = cfg["ollama"]
    model = cfg["models"][role]
    options = {"temperature": 0, "seed": cfg["seed"] if seed is None else seed, "num_ctx": o["num_ctx"]}
    if max_tokens:
        options["num_predict"] = max_tokens
    last_err = None
    for attempt in range(o.get("max_retries", 3)):
        try:
            t0 = time.perf_counter()
            r = ollama_client(cfg).chat(model=model, messages=messages, format=schema,
                                        options=options, keep_alive=o["keep_alive"])
            wall = time.perf_counter() - t0
            content = r.message.content or ""
            in_est = sum(ntok(cfg, m["content"]) for m in messages)
            return {
                "content": content,
                "json": parse_json(content) if schema is not None else None,
                "in_tokens": in_est,                       # tokenizer estimate (prefix cache does not hide it)
                "in_tokens_ollama": r.prompt_eval_count or 0,
                "out_tokens": r.eval_count or 0,
                "wall_s": wall,
                "load_s": (r.load_duration or 0) / 1e9,
                "truncated_risk": in_est >= o["num_ctx"] - 64,
                "model": model,
            }
        except Exception as e:  # network / model load / server errors
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{role} ({model}) failed after retries: {last_err}")


def unload_ollama_models(cfg: dict, except_model: str | None = None) -> None:
    """Explicitly tell Ollama to unload unused models from memory to prevent memory bloat."""
    try:
        import urllib.request
        host = cfg["ollama"]["host"]
        req = urllib.request.Request(f"{host}/api/ps")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        for m in data.get("models", []):
            name = m.get("model") or m.get("name")
            if name and name != except_model:
                data_post = json.dumps({"model": name, "keep_alive": 0}).encode()
                for endpoint in ("/api/chat", "/api/generate"):
                    try:
                        req_post = urllib.request.Request(f"{host}{endpoint}", data=data_post, headers={"Content-Type": "application/json"})
                        urllib.request.urlopen(req_post, timeout=5)
                    except Exception:
                        pass
    except Exception:
        pass


def free_memory() -> None:
    """Collects Python garbage and purges PyTorch MPS allocations."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def warmup(cfg: dict, role: str) -> float:
    """Load the model before timing so load_duration never pollutes latency.
    Unloads any other model first to ensure only one LLM resides in unified memory."""
    target_model = cfg["models"][role]
    unload_ollama_models(cfg, except_model=target_model)
    free_memory()
    r = llm(cfg, role, [{"role": "user", "content": "Reply with OK."}], max_tokens=2)
    return r["load_s"]


# ---------------------------------------------------------------- text helpers
def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def quote_in_text(quote: str, text: str, min_chars: int = 15) -> bool:
    q = norm_ws(quote).strip(" .\"'")
    return len(q) >= min_chars and q in norm_ws(text)


def passage_text(p: dict) -> str:
    return f"{p['header']}\n{p['text']}"


def options_block(q: dict) -> str:
    return "\n".join(f"{i}. {t}" for i, t in enumerate(q["options"], start=1))
