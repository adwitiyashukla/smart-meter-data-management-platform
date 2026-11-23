from __future__ import annotations

import ast
import io
import subprocess
import sys
import tokenize
from pathlib import Path

TEXT_SUFFIXES = {".py", ".toml", ".md", ".yml", ".yaml", ".csv", ".txt", ".cfg", ".gitignore"}
SHEBANG = "#!"


def tracked_files() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return [Path(name) for name in output if name]


def python_findings(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    findings = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and not token.string.startswith(SHEBANG):
            findings.append(f"{path}:{token.start[0]} comment {token.string.strip()[:50]!r}")
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        holder = isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        if holder and ast.get_docstring(node) is not None:
            line = getattr(node, "lineno", 1)
            name = getattr(node, "name", "module")
            findings.append(f"{path}:{line} docstring in {name}")
    return findings


def ascii_findings(path: Path) -> list[str]:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    raw = path.read_bytes()
    findings = []
    for number, line in enumerate(raw.split(b"\n"), start=1):
        try:
            line.decode("ascii")
        except UnicodeDecodeError:
            findings.append(f"{path}:{number} non ascii bytes")
    return findings


def main() -> int:
    findings = []
    for path in tracked_files():
        if not path.is_file():
            continue
        if path.suffix == ".py":
            findings.extend(python_findings(path))
        findings.extend(ascii_findings(path))
    if findings:
        for line in findings:
            print(line)
        print(f"{len(findings)} findings", file=sys.stderr)
        return 1
    print("no comments, no docstrings, ascii only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
