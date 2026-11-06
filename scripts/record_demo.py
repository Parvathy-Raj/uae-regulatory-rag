"""Record a demo of the chat UI: a .webm video plus a README GIF.

Usage (with the app running, e.g. `make ui`):
    python scripts/record_demo.py --url http://localhost:8501 --out docs

Needs the optional demo extras: `pip install -e ".[demo]"` and
`python -m playwright install ffmpeg`. Uses the locally installed Microsoft Edge
(`--channel msedge`) or Chrome (`--channel chrome`), so no browser download.
Everything shown is a live answer from the running app; nothing is staged.
"""

from __future__ import annotations

import argparse
import shutil
import time
from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

QUESTIONS = [
    ("hybrid", "When is a Corporate Tax return due?"),
    ("hybrid", "What conditions must a Free Zone Person meet to be a Qualifying Free Zone Person?"),
    ("hybrid", "What is the fine for parking in a no-parking zone in Dubai?"),
]
MODE_BUTTON = {"dense": "Dense", "hybrid": "Hybrid (BM25 + dense)", "hybrid_rerank": "Hybrid + reranker"}
VIEWPORT = {"width": 1000, "height": 860}


class GifRecorder:
    """Collects screenshots as GIF frames, each with its own display time."""

    def __init__(self, page: Page, width: int = 760):
        self.page, self.width = page, width
        self.frames: list[tuple[Image.Image, int]] = []

    def snap(self, hold_ms: int = 120) -> None:
        img = Image.open(BytesIO(self.page.screenshot())).convert("RGB")
        ratio = self.width / img.width
        img = img.resize((self.width, int(img.height * ratio)), Image.LANCZOS)
        self.frames.append((img, hold_ms))

    def save(self, path: Path) -> None:
        pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f, _ in self.frames]
        pal[0].save(path, save_all=True, append_images=pal[1:], loop=0,
                    duration=[d for _, d in self.frames], optimize=True)


def bot_count(page: Page) -> int:
    return page.locator(".msg.bot").count()


def type_question(page: Page, gif: GifRecorder, text: str) -> None:
    box = page.get_by_placeholder("Ask about VAT or Corporate Tax…")
    box.click()
    step = max(1, len(text) // 12)
    for i in range(0, len(text), step):
        box.press_sequentially(text[i : i + step], delay=35)
        gif.snap(90)
    gif.snap(500)
    box.press("Enter")


def wait_for_answer(page: Page, gif: GifRecorder, before: int, timeout_s: int = 180) -> None:
    deadline = time.time() + timeout_s
    shown_thinking = False
    while time.time() < deadline:
        if bot_count(page) > before and page.locator('[data-testid="stSpinner"]').count() == 0:
            break
        if not shown_thinking and page.locator('[data-testid="stSpinner"]').count():
            page.locator('[data-testid="stSpinner"]').first.scroll_into_view_if_needed()
            gif.snap(900)
            shown_thinking = True
        page.wait_for_timeout(500)
    else:
        raise TimeoutError("No answer within the time limit")
    page.wait_for_timeout(600)
    page.locator(".msg.bot").last.scroll_into_view_if_needed()
    page.mouse.wheel(0, 250)
    page.wait_for_timeout(400)
    gif.snap(3200)


def open_passages(page: Page, gif: GifRecorder) -> None:
    expander = page.locator('[data-testid="stExpander"] summary').last
    if expander.count():
        expander.click()
        page.wait_for_timeout(700)
        page.mouse.wheel(0, 300)
        page.wait_for_timeout(400)
        gif.snap(3000)
        expander.click()
        page.wait_for_timeout(400)


def record(url: str, out: Path, channel: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    video_tmp = out / "_video_tmp"
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, headless=True)
        ctx = browser.new_context(viewport=VIEWPORT, record_video_dir=str(video_tmp),
                                  record_video_size=VIEWPORT)
        page = ctx.new_page()
        page.goto(url)
        page.get_by_placeholder("Ask about VAT or Corporate Tax…").wait_for(timeout=120_000)
        page.wait_for_timeout(1500)
        gif = GifRecorder(page)
        gif.snap(2200)
        current = None
        for i, (mode, question) in enumerate(QUESTIONS):
            if mode != current:
                page.get_by_text(MODE_BUTTON[mode], exact=True).click()
                page.wait_for_timeout(800)
                gif.snap(700)
                current = mode
            before = bot_count(page)
            type_question(page, gif, question)
            wait_for_answer(page, gif, before)
            if i == 1:
                open_passages(page, gif)
        page.wait_for_timeout(1500)
        video_path = Path(page.video.path())
        ctx.close()
        browser.close()
    shutil.move(str(video_path), out / "demo.webm")
    shutil.rmtree(video_tmp, ignore_errors=True)
    gif.save(out / "demo.gif")
    for name in ("demo.webm", "demo.gif"):
        print(f"{name}: {(out / name).stat().st_size / 1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--out", type=Path, default=Path("docs"))
    ap.add_argument("--channel", default="msedge", help="msedge | chrome | chromium")
    args = ap.parse_args()
    record(args.url, args.out, args.channel)


if __name__ == "__main__":
    main()
