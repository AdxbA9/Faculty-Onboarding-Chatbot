"""Acceptance-test launcher (outside the repository): the frozen PF-6 app in ACTIVE mode with guide scopes ON.
The Groq key is loaded privately from the repository's .env; the Plan F switches are set in THIS process only.
Logs the resolved configuration and the PID (never the key), then calls ui.chat.start_app(port=8501).

    python -X utf8 launch_active.py <repo> <log file>
"""
import os
import sys

REPO, LOG = os.path.abspath(sys.argv[1]), sys.argv[2]
os.chdir(REPO)
sys.path.insert(0, REPO)

from dotenv import load_dotenv                                      # noqa: E402
load_dotenv(os.path.join(REPO, ".env"))
os.environ["PLAN_F_ENABLED"] = "1"
os.environ.pop("PLAN_F_SHADOW", None)
os.environ["PLAN_F_GUIDE_SCOPES"] = "1"
# llama-3.3-70b-versatile (the config default) is no longer available to this Groq key (HTTP 404); the project owner chose
# openai/gpt-oss-120b for the acceptance test. config.GROQ_MODEL reads this variable; nothing in the repository changes.
os.environ["GROQ_MODEL"] = "openai/gpt-oss-120b"
# D-117: a reasoning model's hidden tokens count against the answer cap; give it an allowance (0 = PF-6 behaviour).
os.environ["LLM_REASONING_TOKENS"] = "2048"

from handbook_bot import config                                     # noqa: E402
from ui import pipeline                                             # noqa: E402

key = os.getenv("GROQ_API_KEY", "").strip()
with open(LOG, "w", encoding="utf-8") as f:
    f.write("pid=%d\n" % os.getpid())
    f.write("mode=%s\n" % pipeline.plan_f_mode())
    f.write("PLAN_F_ENABLED=%s PLAN_F_SHADOW=%s PLAN_F_GUIDE_SCOPES=%s\n"
            % (config.PLAN_F_ENABLED, config.PLAN_F_SHADOW, config.PLAN_F_GUIDE_SCOPES))
    f.write("GROQ_MODEL=%s LLM_REASONING_TOKENS=%s\n" % (config.GROQ_MODEL, config.LLM_REASONING_TOKENS))
    f.write("groq_is_configured=%s key_length=%d key_prefix_ok=%s\n" % (pipeline.groq_is_configured(), len(key), key.startswith("gsk_")))
del key

from ui.chat import start_app                                       # noqa: E402

start_app(port=8501)
