"""PF-7 validation through the real seam in the acceptance configuration (ACTIVE, guide scopes ON, the Groq model and the
reasoning allowance from the environment, key loaded privately from .env). The production code runs unchanged: only the
Groq client is wrapped to record each response's finish reason and token use. Never prints the key.

    python -X utf8 probe2.py <repo> <questions file> <out json>
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
REPO = os.path.abspath(sys.argv[1])
os.chdir(REPO)
sys.path.insert(0, REPO)
from dotenv import load_dotenv                                      # noqa: E402
load_dotenv(os.path.join(REPO, ".env"))
os.environ.setdefault("GROQ_MODEL", "openai/gpt-oss-120b")
os.environ.setdefault("LLM_REASONING_TOKENS", "2048")
os.environ["PLAN_F_ENABLED"] = "1"
os.environ.pop("PLAN_F_SHADOW", None)
os.environ["PLAN_F_GUIDE_SCOPES"] = "1"

from handbook_bot import config                                     # noqa: E402
from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402
from ui import pipeline, presentation                               # noqa: E402

CALLS = []


class Recorder:
    def __init__(self, inner):
        self._inner = inner

    def with_options(self, **kw):
        return Recorder(self._inner.with_options(**kw))

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kw):
        resp = self._inner.chat.completions.create(**kw)
        d = getattr(resp.usage, "completion_tokens_details", None)
        CALLS.append({"max_tokens": kw.get("max_tokens"), "finish": resp.choices[0].finish_reason, "tokens": resp.usage.completion_tokens,
                      "reasoning": getattr(d, "reasoning_tokens", None) if d else None})
        return resp


questions = [l.strip() for l in open(sys.argv[2], encoding="utf-8") if l.strip() and not l.startswith("#")]
kb = build_knowledge_base(verbose=False)
kb.groq_client = Recorder(kb.groq_client)
print("model", config.GROQ_MODEL, "| reasoning allowance", config.LLM_REASONING_TOKENS, "| mode", pipeline.plan_f_mode(),
      "| scopes", config.PLAN_F_GUIDE_SCOPES)
out = []
for q in questions:
    CALLS.clear()
    res, d = pipeline.ask_with_diagnostics(kb, q)
    trace = [(t.agent, t.decision, {k: v for k, v in (t.detail or {}).items() if k in ("best_score", "reason", "error", "topics", "chunks_removed",
                                                                                     "chunks_screened")})
             for t in (getattr(res, "agent_trace", None) or [])
             if t.agent in ("router", "evidence", "synthesis", "verifier", "conflict_guard", "plan_f_outcome")]
    rec = {"question": q, "served": (d or {}).get("served"), "outcome": (d or {}).get("outcome"), "reason": ((d or {}).get("reason") or "")[:140],
           "answer": res.answer, "pages": list(res.pages), "chips": presentation.source_chips(res),
           "items": [((i.get("meta") or {}).get("source_id"), (i.get("meta") or {}).get("page"), (i.get("chunk") or "")[:160]) for i in res.items],
           "llm": list(CALLS), "trace": trace}
    out.append(rec)
    print("\n=== " + q)
    print("served:", rec["served"], "|", rec["outcome"], "|", rec["reason"][:90])
    print("answer:", res.answer[:420].replace("\n", " / "))
    print("chips:", rec["chips"])
    print("llm:", [(c["finish"], c["tokens"], c["reasoning"]) for c in rec["llm"]], "| trace:", [(a, b) for a, b, _ in trace])
json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
