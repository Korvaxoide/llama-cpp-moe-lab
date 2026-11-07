#!/usr/bin/env python3
"""Validazione statica della repo (usata dalla CI e in locale).

Controlla, senza dipendenze esterne:
  1. ogni `results/*.jsonl` è JSON valido;
  2. ogni riga di `bench_*.jsonl` ha `n_ctx == 200192` (requisito del laboratorio);
  3. `results/summary.csv` ha l'intestazione attesa e almeno una riga;
  4. ogni link relativo nei file Markdown punta a un file esistente;
  5. tutti i file Python compilano;
  6. non ci sono path personali assoluti (`/home/...`) nei file versionati.

Uso: python3 scripts/check_repo.py
Exit code 0 se tutto passa, 1 altrimenti.
"""
from __future__ import annotations

import csv
import json
import py_compile
import re
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
EXPECTED_CTX = 200192
SUMMARY_HEADER = [
    "tag", "depth", "n", "pp_tps", "tg_tps", "tg_min", "tg_max",
    "acc", "ttft_s", "vram_max", "ram_avail_min", "n_ctx",
]
SKIP_DIRS = {".git", "src", "__pycache__", "node_modules"}
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def fail(msg: str) -> None:
    print(f"FAIL  {msg}")


def ok(msg: str) -> None:
    print(f"ok    {msg}")


def walk(pattern: str) -> list[Path]:
    out = []
    for p in LAB.rglob(pattern):
        if not any(part in SKIP_DIRS for part in p.parts):
            out.append(p)
    return sorted(out)


def check_jsonl() -> int:
    errors = 0
    files = sorted((LAB / "results").glob("*.jsonl"))
    if not files:
        fail("nessun file results/*.jsonl")
        return 1
    for path in files:
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            fail(f"{path.relative_to(LAB)}: JSON non valido ({exc})")
            errors += 1
            continue
        if not rows:
            fail(f"{path.relative_to(LAB)}: vuoto")
            errors += 1
            continue
        if path.name.startswith("bench_"):
            bad = [r.get("n_ctx") for r in rows if r.get("n_ctx") != EXPECTED_CTX]
            if bad:
                fail(f"{path.relative_to(LAB)}: n_ctx inatteso {sorted(set(bad))}")
                errors += 1
        ok(f"{path.relative_to(LAB)}: {len(rows)} righe")
    return errors


def check_summary() -> int:
    path = LAB / "results" / "summary.csv"
    if not path.exists():
        fail("results/summary.csv manca")
        return 1
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows or rows[0] != SUMMARY_HEADER:
        fail(f"results/summary.csv: intestazione inattesa {rows[0] if rows else 'vuoto'}")
        return 1
    if len(rows) < 2:
        fail("results/summary.csv: nessuna riga di dati")
        return 1
    ok(f"results/summary.csv: {len(rows) - 1} righe di dati")
    return 0


def check_links() -> int:
    errors = 0
    files = walk("*.md")
    for path in files:
        text = path.read_text(encoding="utf-8")
        for regex in (LINK_RE, IMG_RE):
            for target in regex.findall(text):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target = target.split("#", 1)[0]
                if not target:
                    continue
                resolved = (path.parent / target).resolve()
                if not resolved.exists():
                    fail(f"{path.relative_to(LAB)}: link rotto -> {target}")
                    errors += 1
    if not errors:
        ok(f"link relativi: {len(files)} file Markdown")
    return errors


def check_python() -> int:
    errors = 0
    files = walk("*.py")
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            fail(f"{path.relative_to(LAB)}: non compila ({exc.msg})")
            errors += 1
    if not errors:
        ok(f"Python: {len(files)} file compilano")
    return errors


def check_private_paths() -> int:
    errors = 0
    suspects = ("/home/", "/Users/")
    myself = Path(__file__).resolve()
    for path in walk("*"):
        if not path.is_file() or path.suffix in {".gguf", ".png", ".jpg"}:
            continue
        if path.resolve() == myself:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for needle in suspects:
            if needle in text:
                fail(f"{path.relative_to(LAB)}: path personale '{needle}'")
                errors += 1
                break
    if not errors:
        ok("nessun path personale assoluto")
    return errors


def main() -> int:
    print(f"check_repo: {LAB}\n")
    errors = 0
    errors += check_jsonl()
    errors += check_summary()
    errors += check_links()
    errors += check_python()
    errors += check_private_paths()
    print()
    if errors:
        print(f"{errors} problema/i")
        return 1
    print("tutto ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
