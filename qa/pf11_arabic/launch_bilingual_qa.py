"""Launcher for the bilingual browser QA (post-freeze PF-11): the app in ACTIVE mode with guide scopes on, the Groq key
loaded privately from the repository's .env so the header shows the real service status, and the Arabic model requests
switched OFF in THIS process only (``ARABIC_ANSWER_TRANSLATION=0``, ``ARABIC_QUERY_MODEL_FALLBACK=0``), so the browser run
makes no inference request: Arabic answers are shown as the Arabic lead-in plus the verified English text. A process cap
of 0 requests is set as a second guard. Logs the resolved configuration and the PID (never the key).

    python -X utf8 qa/pf11_arabic/launch_bilingual_qa.py <repo> <log file> [--translate]

``--translate`` keeps the translation ON with a process cap of 2 requests (two screenshot cases of the authorised live
budget); without it the cap is 0.
"""
import os
import sys

REPO, LOG = os.path.abspath(sys.argv[1]), sys.argv[2]
TRANSLATE = "--translate" in sys.argv[3:]
os.chdir(REPO)
sys.path.insert(0, REPO)

from dotenv import load_dotenv                                      # noqa: E402
load_dotenv(os.path.join(REPO, ".env"))
os.environ["PLAN_F_ENABLED"] = "1"
os.environ.pop("PLAN_F_SHADOW", None)
os.environ["PLAN_F_GUIDE_SCOPES"] = "1"
os.environ.setdefault("GROQ_MODEL", "openai/gpt-oss-120b")
os.environ.setdefault("LLM_REASONING_TOKENS", "2048")
os.environ["ARABIC_QUERY_MODEL_FALLBACK"] = "0"
os.environ["ARABIC_ANSWER_TRANSLATION"] = "1" if TRANSLATE else "0"
os.environ["LLM_PROCESS_CALL_CAP"] = "2" if TRANSLATE else "1"      # 1: the startup model check may use one request; no inference follows

from handbook_bot import config                                     # noqa: E402
from ui import pipeline                                             # noqa: E402

key = os.getenv("GROQ_API_KEY", "").strip()
with open(LOG, "w", encoding="utf-8") as f:
    f.write("pid=%d\n" % os.getpid())
    f.write("mode=%s\n" % pipeline.plan_f_mode())
    f.write("PLAN_F_ENABLED=%s PLAN_F_SHADOW=%s PLAN_F_GUIDE_SCOPES=%s\n" % (config.PLAN_F_ENABLED, config.PLAN_F_SHADOW, config.PLAN_F_GUIDE_SCOPES))
    f.write("ARABIC_SUPPORT=%s ARABIC_ANSWER_TRANSLATION=%s ARABIC_QUERY_MODEL_FALLBACK=%s LLM_PROCESS_CALL_CAP=%s\n"
            % (config.ARABIC_SUPPORT, config.ARABIC_ANSWER_TRANSLATION, config.ARABIC_QUERY_MODEL_FALLBACK, config.LLM_PROCESS_CALL_CAP))
    f.write("GROQ_MODEL=%s LLM_REASONING_TOKENS=%s\n" % (config.GROQ_MODEL, config.LLM_REASONING_TOKENS))
    f.write("groq_is_configured=%s key_length=%d key_prefix_ok=%s\n" % (pipeline.groq_is_configured(), len(key), key.startswith("gsk_")))
del key

from ui.chat import start_app                                       # noqa: E402

start_app(port=8501)
