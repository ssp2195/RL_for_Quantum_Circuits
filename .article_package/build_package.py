from __future__ import annotations

import base64
import gzip
import hashlib
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / ".article_package" / "source"
OUT = ROOT / "research_package"
MD_NAME = "Article_limited_scope_4_revised_persistent_DAG_RL.md"
DOCX_NAME = "Article_limited_scope_4_revised_persistent_DAG_RL.docx"
TEX_NAME = "Article_limited_scope_4_revised_persistent_DAG_RL.tex"
ZIP_NAME = "Article_limited_scope_4_revised_package.zip"
EXPECTED_MD_SHA256 = "887cc62d61b43bc0470d99d8d770eea37052633f44cc1975eaf4bf4202871ae4"

parts = sorted(SOURCE_DIR.glob("article.md.gz.b64.part*"))
if not parts:
    raise SystemExit("No manuscript source parts found")
encoded = "".join(p.read_text(encoding="utf-8").strip() for p in parts)
markdown_bytes = gzip.decompress(base64.b64decode(encoded))
actual = hashlib.sha256(markdown_bytes).hexdigest()
if actual != EXPECTED_MD_SHA256:
    raise SystemExit(f"Markdown SHA256 mismatch: {actual}")

OUT.mkdir(parents=True, exist_ok=True)
md = OUT / MD_NAME
docx = OUT / DOCX_NAME
tex = OUT / TEX_NAME
archive = OUT / ZIP_NAME
md.write_bytes(markdown_bytes)

subprocess.run(["pandoc", str(md), "-o", str(docx)], check=True)
subprocess.run(["pandoc", str(md), "-s", "-o", str(tex)], check=True)

with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(md, arcname=MD_NAME)
    zf.write(docx, arcname=DOCX_NAME)
    zf.write(tex, arcname=TEX_NAME)

for path in (md, docx, tex, archive):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{path.name}\t{path.stat().st_size}\t{digest}")
