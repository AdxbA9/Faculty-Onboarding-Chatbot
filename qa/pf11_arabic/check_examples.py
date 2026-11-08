"""Which Arabic first-use examples are answered WITHOUT a model call, like their English counterparts (post-freeze PF-11).

    python -X utf8 qa/pf11_arabic/check_examples.py <repo> <out json>

ACTIVE mode, guide scopes on, no key: every Arabic example of ``ui.i18n.TOPICS_AR`` and every English example of
``ui.chat.TOPICS`` go through the seam; an example is OK when the specialist path served it (or the extractor did) and the
Arabic one reached the same source pages as its English pair."""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
REPO = os.path.abspath(sys.argv[1])
os.chdir(REPO)
sys.path.insert(0, REPO)
for name in ("GROQ_API_KEY", "PLAN_F_SHADOW"):
    os.environ.pop(name, None)
os.environ["PLAN_F_ENABLED"] = "1"
os.environ["PLAN_F_GUIDE_SCOPES"] = "1"
from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402
from ui import chat, i18n, pipeline                                 # noqa: E402

kb = build_knowledge_base(verbose=False)
rows = []
for en_topic, ar_topic in zip(chat.TOPICS, i18n.TOPICS_AR):
    for en_q, ar_q in zip(en_topic["examples"], ar_topic["examples"]):
        en_r, en_d = pipeline.ask_with_diagnostics(kb, en_q)
        ar_r, ar_d = pipeline.ask_with_diagnostics(kb, ar_q)
        def ok(r, d):
            return d.get("served") == "plan_f" or (not r.used_llm and "GROQ_API_KEY" not in (r.original_answer or r.answer) and not (r.original_answer or r.answer).startswith("I do not have"))
        rows.append({"topic": en_topic["id"], "en": en_q, "ar": ar_q, "en_served": en_d.get("served"), "ar_served": ar_d.get("served"),
                     "en_pages": en_r.pages, "ar_pages": ar_r.pages, "english_query": ar_r.query_transform.get("english_query"),
                     "ar_presentation": ar_r.translation.get("status"), "en_ok": ok(en_r, en_d), "ar_ok": ok(ar_r, ar_d),
                     "same_pages": en_r.pages == ar_r.pages, "same_english_text": ar_r.original_answer == en_r.answer})
        print("%-18s en_ok=%s ar_ok=%s same_pages=%s same_text=%s | %s" % (en_topic["id"], rows[-1]["en_ok"], rows[-1]["ar_ok"], rows[-1]["same_pages"], rows[-1]["same_english_text"], ar_q))
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump({"mode": pipeline.plan_f_mode(), "rows": rows, "all_ok": all(r["ar_ok"] and r["same_pages"] for r in rows)}, f, indent=1, ensure_ascii=False)
print("all_ok:", all(r["ar_ok"] and r["same_pages"] for r in rows))
