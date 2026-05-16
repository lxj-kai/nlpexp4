"""校验中期检查报告的图片引用 / 结构完整性。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
ROOT = Path(__file__).resolve().parent.parent
html_path = ROOT / "report" / "中期检查报告.html"
html = html_path.read_text(encoding="utf-8")

imgs = re.findall(r'<img[^>]*src="([^"]+)"', html)
print("[images]")
for src in imgs:
    p = (html_path.parent / src).resolve()
    print("  exist=", p.exists(), "|", src)

secs = re.findall(r'class="sec-title">([^<]+)<', html)
print(f"\n[sections] {len(secs)}")
for s in secs:
    print("  -", s)

print(f"\n[stats] {len(html):,} chars")
