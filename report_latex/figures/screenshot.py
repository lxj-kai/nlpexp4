from playwright.sync_api import sync_playwright
import os

base = r"D:\code\nlpexp4\report_latex\figures"

with sync_playwright() as p:
    browser = p.chromium.launch()

    arch_path = os.path.join(base, "system_architecture.html")
    url = "file:///" + arch_path.replace("\\", "/")
    print(f"Loading: {url}")

    page = browser.new_page(viewport={"width": 1400, "height": 820})
    resp = page.goto(url, wait_until="domcontentloaded")
    print(f"Status: {resp.status if resp else 'None'}")
    page.wait_for_timeout(3000)

    html_len = page.evaluate("document.body.innerHTML.length")
    print(f"Body innerHTML length: {html_len}")

    if html_len == 0:
        content = page.content()
        print(f"Full page content length: {len(content)}")
        print(f"First 200 chars: {content[:200]}")

    page.screenshot(
        path=os.path.join(base, "system_architecture.png"),
        full_page=True,
    )
    sz = os.path.getsize(os.path.join(base, "system_architecture.png"))
    print(f"Arch screenshot: {sz} bytes")

    page2 = browser.new_page(viewport={"width": 1200, "height": 500})
    eng_path = os.path.join(base, "data_engine.html")
    url2 = "file:///" + eng_path.replace("\\", "/")
    page2.goto(url2, wait_until="domcontentloaded")
    page2.wait_for_timeout(2000)
    page2.screenshot(
        path=os.path.join(base, "data_engine.png"),
        full_page=True,
    )
    sz2 = os.path.getsize(os.path.join(base, "data_engine.png"))
    print(f"Engine screenshot: {sz2} bytes")

    browser.close()
    print("Done")
