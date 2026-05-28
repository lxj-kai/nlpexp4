"""Restore architecture HTML from git and regenerate PNGs."""
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = Path(r"D:\code\nlpexp4\report_latex\figures")
REPO = Path(r"D:\code\nlpexp4")


def git_file(rev_path: str) -> str:
    raw = subprocess.check_output(
        ["git", "show", rev_path],
        cwd=str(REPO),
    )
    return raw.decode("utf-8")


# Restore original architecture diagram (user requested previous version)
arch_html = git_file("aa9598f:report_latex/figures/system_architecture.html")
(BASE / "system_architecture.html").write_text(arch_html, encoding="utf-8", newline="\n")
print(f"Restored architecture HTML: {len(arch_html)} chars")


def screenshot(html_name: str, png_name: str, viewport: dict, selector: str):
    html = BASE / html_name
    out = BASE / png_name
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=viewport)
        page.goto(html.as_uri(), wait_until="domcontentloaded")
        page.wait_for_timeout(2000)
        body_len = page.evaluate("document.body.innerHTML.length")
        print(f"{html_name}: body innerHTML = {body_len}")
        if body_len == 0:
            raise RuntimeError(f"{html_name} rendered empty")
        page.locator(selector).screenshot(path=str(out))
        print(f"{png_name} -> {out.stat().st_size} bytes")
        browser.close()


screenshot(
    "system_architecture.html",
    "system_architecture.png",
    {"width": 1400, "height": 820},
    "main.canvas",
)
screenshot(
    "data_engine.html",
    "data_engine.png",
    {"width": 1200, "height": 500},
    ".canvas",
)
print("Done")
