"""Headless smoke test of app.py: render, click sample questions, print the answers.

Usage: python scripts/smoke_app.py [question-index ...]
Needs a working LLM provider (Ollama running locally, or LLM_PROVIDER=groq with GROQ_API_KEY).
"""

import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def run(indices: list[int]) -> int:
    failures = 0
    for i in indices:
        at = AppTest.from_file(APP, default_timeout=900)
        at.run()
        assert not at.exception, at.exception
        print("caption:", at.caption[0].value)
        print("sidebar:", " | ".join(m.value for m in at.sidebar.markdown))

        buttons = [b for b in at.button if b.label != "Clear chat"]
        question = buttons[i].label
        start = time.time()
        buttons[i].click().run()
        elapsed = time.time() - start

        if at.exception or at.error:
            failures += 1
            print(f"\nQ{i}: {question}\n  FAILED: {[e.value for e in at.error] or at.exception}")
            continue
        answer = at.chat_message[-1].markdown[0].value
        status = [x.value for x in (*at.info, *at.success, *at.warning)]
        sources = at.expander[-1].label if at.expander else "-"
        print(f"\nQ{i}: {question} ({elapsed:.0f}s)\n  status: {status}\n  {sources}\n  answer: {answer}")
    return failures


if __name__ == "__main__":
    picks = [int(a) for a in sys.argv[1:]] or [0, 3]
    sys.exit(1 if run(picks) else 0)
