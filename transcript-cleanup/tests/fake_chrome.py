#!/usr/bin/env python3
"""Поддельный `google-chrome` для тестов reader_pdf: вместо печати пишет в
--print-to-pdf заглушку с исходным HTML внутри, чтобы тест мог проверить, что
печаталось.

FAKE_CHROME_FAIL=1 — печать падает (rc 1, PDF не создаётся).
"""
import os
import sys
from pathlib import Path
from urllib.parse import unquote


def main() -> int:
    if os.environ.get("FAKE_CHROME_FAIL") == "1":
        print("simulated chrome crash", file=sys.stderr)
        return 1
    out = next(a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--print-to-pdf="))
    url = sys.argv[-1]
    assert url.startswith("file://"), url
    html = Path(unquote(url[len("file://"):])).read_text(encoding="utf-8")
    Path(out).write_text("%PDF-fake\n" + html, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
