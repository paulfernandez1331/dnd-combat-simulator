"""Drive board/index.html in Edge.

Build a spec with ``battle_payload`` and pass it to ``BoardPage.show``. The page
is the real mat, so a click is a real square or wheel button. Use it from
``tests/test_battlemap.py`` or a debug session:

    python -m pytest -q tests/test_battlemap.py
"""

from __future__ import annotations

from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from playwright.sync_api import sync_playwright

from src.paths import ROOT

BOARD_DIR = ROOT / "board"
_LISTEN = """
window.__boardMessages = [];
window.addEventListener("message", (event) => {
  const data = event.data;
  if (data && data.isStreamlitMessage) window.__boardMessages.push(data);
});
"""


class _QuietHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BOARD_DIR), **kwargs)

    def log_message(self, format, *args):
        return


class BoardPage:
    """One loaded mat. ``show`` replaces the spec the way Streamlit does."""

    def __init__(self, page):
        self.page = page

    def show(self, spec: dict) -> None:
        self.page.reload(wait_until="domcontentloaded")
        self.page.locator("#north").wait_for()
        self.page.evaluate(
            """(spec) => {
                window.postMessage(
                    { type: "streamlit:render", args: { spec: spec }, disabled: false },
                    "*"
                );
            }""",
            spec,
        )
        self.page.locator("button.cell").first.wait_for()

    def click(self, name: str) -> None:
        button = self.page.get_by_role("button", name=name, exact=True)
        action = button.get_attribute("data-action")
        before = self._value_count()
        button.click()
        if action in {"ability", "slot", "clear"}:
            self._wait_for_value(before)

    def click_cell(self, x: int, y: int) -> None:
        before = self._value_count()
        self.page.locator(f'button.cell[data-x="{x}"][data-y="{y}"]').click()
        self._wait_for_value(before)

    def _value_count(self) -> int:
        return self.page.evaluate(
            """() => window.__boardMessages.filter(
                (message) => message.type === "streamlit:setComponentValue"
            ).length"""
        )

    def _wait_for_value(self, before: int) -> None:
        # postMessage is delivered on the turn after the click.
        self.page.wait_for_function(
            """(before) => window.__boardMessages.filter(
                (message) => message.type === "streamlit:setComponentValue"
            ).length > before""",
            arg=before,
        )

    def values(self) -> list[dict]:
        return self.page.evaluate(
            """() => window.__boardMessages
                .filter((message) => message.type === "streamlit:setComponentValue")
                .map((message) => message.value)"""
        )

    def clear(self) -> None:
        self.page.evaluate("() => { window.__boardMessages = []; }")

    def hint(self) -> str:
        return self.page.locator("#hint").inner_text()

    def hit_at(self, x: int, y: int) -> dict:
        """What a click in the middle of a square actually lands on."""
        return self.page.evaluate(
            """([x, y]) => {
                const cell = document.querySelector(`button.cell[data-x="${x}"][data-y="${y}"]`);
                const rect = cell.getBoundingClientRect();
                const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
                return {
                    covered: Boolean(hit && hit.closest("#wheel")),
                    label: hit ? (hit.getAttribute("aria-label") || hit.textContent || "").trim() : "",
                };
            }""",
            [x, y],
        )


@contextmanager
def open_board():
    """Launch headless Edge against the mat. Close it when the block ends."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _QuietHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="msedge", headless=True)
            context = browser.new_context(viewport={"width": 1100, "height": 900})
            page = context.new_page()
            page.set_default_timeout(5000)
            page.add_init_script(_LISTEN)
            page.goto(f"http://{host}:{port}/index.html", wait_until="domcontentloaded")
            try:
                yield BoardPage(page)
            finally:
                context.close()
                browser.close()
    finally:
        server.shutdown()
