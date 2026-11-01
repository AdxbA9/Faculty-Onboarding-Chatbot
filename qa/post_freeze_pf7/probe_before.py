"""PF-7 instrumented probe used BEFORE the D-117 change (the reproduction): every legacy synthesis call's finish reason,
reasoning tokens and answer length. The call replays the PF-6 ``qa.ask_groq`` arguments (cap ``LLM_MAX_TOKENS`` = 512, no
reasoning allowance) only to read the response metadata; the D-117 validation uses ``validate_live.py`` instead.

    python -X utf8 probe.py <repo> <questions file> <out json>
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
os.environ["PLAN_F_ENABLED"] = "1"
os.environ.pop("PLAN_F_SHADOW", None)
os.environ["PLAN_F_GUIDE_SCOPES"] = "1"

from handbook_bot import config, qa                                 # noqa: E402
from handbook_bot.agents import synthesis                           # noqa: E402
from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402
from ui import pipeline, presentation                               # noqa: E402

CALLS = []


def ask_groq_recorded(client, prompt):
    caller = client.with_options(max_retries=max(0, config.LLM_TRANSPORT_RETRIES))
    resp = caller.chat.completions.create(
        model=config.GROQ_MODEL, temperature=config.LLM_TEMPERATURE, max_tokens=config.LLM_MAX_TOKENS,
        messages=[{"role": "system", "content": "You answer accurately from supplied context only. Be conservative. Never invent facts."},
                  {"role": "user", "content": prompt}])
    ch = resp.choices[0]
    d = getattr(resp.usage, "completion_tokens_details", None)
    content = (ch.message.content or "").strip()
    CALLS.append({"finish_reason": ch.finish_reason, "completion_tokens": resp.usage.completion_tokens,
                  "reasoning_tokens": getattr(d, "reasoning_tokens", None) if d else None, "content_chars": len(content)})
    return content


synthesis.ask_groq = ask_groq_recorded
questions = [l.strip() for l in open(sys.argv[2], encoding="utf-8") if l.strip() and not l.startswith("#")]
kb = build_knowledge_base(verbose=False)
print("model", config.GROQ_MODEL, "| mode", pipeline.plan_f_mode(), "| scopes", config.PLAN_F_GUIDE_SCOPES)
out = []
for q in questions:
    CALLS.clear()
    res, d = pipeline.ask_with_diagnostics(kb, q)
    trace = [(t.agent, t.decision, {k: v for k, v in (t.detail or {}).items() if k in ("best_score", "reason", "error", "claimed_pages", "refused")})
             for t in (getattr(res, "agent_trace", None) or []) if t.agent in ("router", "evidence", "synthesis", "verifier", "coordinator", "plan_f_outcome",
                                                                                "guide_inference", "aggregator")]
    rec = {"question": q, "served": (d or {}).get("served"), "outcome": (d or {}).get("outcome"), "reason": ((d or {}).get("reason") or "")[:140],
           "answer": res.answer, "pages": list(res.pages), "chips": presentation.source_chips(res),
           "items": [((i.get("meta") or {}).get("source_id"), (i.get("meta") or {}).get("page")) for i in res.items], "llm": list(CALLS), "trace": trace}
    out.append(rec)
    print("\n=== " + q)
    print("served:", rec["served"], "|", rec["outcome"], "|", rec["reason"][:100])
    print("answer:", res.answer[:300].replace("\n", " / "))
    print("chips:", rec["chips"])
    print("llm:", rec["llm"])
    print("trace:", [(a, b) for a, b, _ in trace], [c.get("best_score") for a, b, c in trace if a == "evidence"])
json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
