# corpus.py
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"

CORPUS_DIR.mkdir(parents=True, exist_ok=True)

stdlib = Path(os.__file__).parent
files = sorted(stdlib.glob("*.py"))
code = "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in files)
(CORPUS_DIR / "corpus_code.txt").write_text(code, encoding="utf-8")
print(f"corpus_code.txt: {len(files)} files, {len(code):,} chars")
 
doc_files = sorted((CORPUS_DIR / Path("python_docs")).rglob("*.txt"))
docs = "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in doc_files)
(CORPUS_DIR / Path("corpus_prose.txt")).write_text(docs, encoding="utf-8")
print(f"corpus_prose.txt: {len(doc_files)} files, {len(docs):,} chars")