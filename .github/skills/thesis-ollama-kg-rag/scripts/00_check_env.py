"""Gate 0/1: verify Python packages, hardware, Ollama server and the four model roles; record the environment.

Usage: python 00_check_env.py [--pull]
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import platform
import subprocess
import sys

from common import llm, load_config, ollama_client, work, write_json
from prompts import GEN_SCHEMA, PROMPT_VERSION

PKGS = ["ollama", "sentence-transformers", "transformers", "torch", "faiss-cpu", "rank-bm25", "networkx",
        "numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "matplotlib", "ragas", "langchain-ollama"]
ROLES = ["generator", "verifier", "evaluator", "kg_extractor"]


def sysctl(key: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return "n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true", help="pull missing Ollama models")
    args = ap.parse_args()
    cfg = load_config()
    problems: list[str] = []

    versions = {}
    for p in PKGS:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
            problems.append(f"missing package {p}")

    import torch
    mem = sysctl("hw.memsize")
    hw = {
        "platform": platform.platform(), "python": sys.version.split()[0], "machine": platform.machine(),
        "cpu": sysctl("machdep.cpu.brand_string"),
        "ram_gb": round(int(mem) / 2**30, 1) if mem.isdigit() else None,
        "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available(),
    }
    if cfg["encoders"]["device"] == "mps" and not hw["mps"]:
        problems.append("encoders.device=mps but MPS unavailable")

    try:
        listed = {m.model: m for m in ollama_client(cfg).list().models}
    except Exception as e:  # server not running
        print(f"Ollama server unreachable at {cfg['ollama']['host']}: {e}\n"
              "Start it with `brew services start ollama` or `ollama serve` (separate terminal).")
        sys.exit(1)

    def present(name: str) -> bool:
        return name in listed or f"{name}:latest" in listed

    wanted = [cfg["models"][r] for r in ROLES] + [cfg["models"]["ragas_embeddings"]]
    for name in wanted:
        if not present(name):
            if args.pull:
                print(f"pulling {name} ...")
                ollama_client(cfg).pull(name)
            else:
                problems.append(f"model not pulled: ollama pull {name}")
    if args.pull:
        listed = {m.model: m for m in ollama_client(cfg).list().models}

    names = [cfg["models"][r] for r in ROLES]
    if len(set(names)) < 4:
        problems.append("role models are not all distinct (thesis requires independent verifier/evaluator)")

    roles = {}
    for role in ROLES:
        name = cfg["models"][role]
        m = listed.get(name) or listed.get(f"{name}:latest")
        if m is None:
            continue
        info = {"model": name, "digest": m.digest, "size_gb": round((m.size or 0) / 1e9, 2),
                "family": m.details.family if m.details else None,
                "parameter_size": m.details.parameter_size if m.details else None,
                "quantization": m.details.quantization_level if m.details else None}
        try:
            r = llm(cfg, role, [{"role": "user", "content": "Question: Which 5GC function handles registration?\n"
                                 "Options:\n1. AMF\n2. UPF\nReturn JSON."}], schema=GEN_SCHEMA, max_tokens=128)
            info.update(smoke_ok=r["json"] is not None, smoke_wall_s=round(r["wall_s"], 2), load_s=round(r["load_s"], 2))
            if r["json"] is None:
                problems.append(f"{role}: structured JSON output failed")
        except RuntimeError as e:
            info.update(smoke_ok=False, error=str(e))
            problems.append(f"{role}: {e}")
        roles[role] = info
    fams = [v.get("family") for v in roles.values()]
    if len(set(fams)) < len(fams):
        print(f"WARNING: role model families overlap {fams}; self-preference bias risk, document it.")

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    work(cfg, "env", "requirements.lock").write_text(freeze)
    write_json(work(cfg, "env", "environment.json"), {
        "hardware": hw, "packages": versions, "ollama_roles": roles, "prompt_version": PROMPT_VERSION,
        "config_path": cfg["_config_path"], "config_sha256": cfg["_config_sha256"], "problems": problems})

    print(f"hardware: {hw}")
    for role, info in roles.items():
        print(f"{role:13s} {info['model']:24s} {info.get('family')} {info.get('quantization')} smoke_ok={info.get('smoke_ok')}")
    if problems:
        print("\nPROBLEMS:\n- " + "\n- ".join(problems))
        sys.exit(2)
    print("\nGate 0/1 passed. Environment recorded in work/env/.")


if __name__ == "__main__":
    main()
