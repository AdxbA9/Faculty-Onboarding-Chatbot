"""PF-7 citation validation on real answers (acceptance configuration, key loaded privately from .env). For every model
answer the orchestrator cites, the SAME answer and evidence give two chip sets:
  * before: the PF-6 logic (every evidence item, the Verifier's page-number citation);
  * after:  the D-117 logic (the result as served: one best supporting item per claim).
A displayed document is "supported" when one of its items on a displayed page supports at least one answer claim
(the Verifier's own item-level support test). Writes JSON; never prints the key.

    python -X utf8 citations.py <repo> <questions file> <out json>
"""
import json
import os
import sys
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")
REPO = os.path.abspath(sys.argv[1])
os.chdir(REPO)
sys.path.insert(0, REPO)
from dotenv import load_dotenv                                      # noqa: E402
load_dotenv(os.path.join(REPO, ".env"))
os.environ.setdefault("GROQ_MODEL", "openai/gpt-oss-120b")
os.environ.setdefault("LLM_REASONING_TOKENS", "2048")
os.environ["PLAN_F_ENABLED"] = "1"
os.environ["PLAN_F_GUIDE_SCOPES"] = "1"

from handbook_bot import orchestrator                               # noqa: E402
from handbook_bot.agents import verifier as V                       # noqa: E402
from handbook_bot.agents.types import EvidenceResult, SynthesisResult  # noqa: E402
from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402
from ui import pipeline, presentation                               # noqa: E402

SEEN = {}
real = orchestrator.citing_items


def recording(answer, items, claimed_pages=()):
    SEEN.update(answer=answer, items=list(items), claimed=list(claimed_pages or ()))
    return real(answer, items, claimed_pages)


orchestrator.citing_items = recording


def title(item):
    return (item.get("meta") or {}).get("source_title")


def supported_docs(answer, items, pages):
    claims = [c for c in (V._Claim(t) for t in V.split_claims(answer)) if c.checkable]
    vitems = [V._Item(i, it) for i, it in enumerate(items)]
    docs = set()
    for claim in claims:
        sup = [vi for vi in vitems if V._support(claim, vi, vitems)[0]] or V._union_support(claim, vitems)
        docs |= {title(items[vi.index]) for vi in sup if vi.page in pages}
    return docs


def doc_chips(chips):
    return [label for cls, label in chips if cls == "doc"]


def short(t):
    return presentation._shorten(t, 40) if t else t


questions = [l.strip() for l in open(sys.argv[2], encoding="utf-8") if l.strip() and not l.startswith("#")]
kb = build_knowledge_base(verbose=False)
out = []
for q in questions:
    SEEN.clear()
    res, d = pipeline.ask_with_diagnostics(kb, q)
    rec = {"question": q, "served": (d or {}).get("served"), "answer": res.answer[:300]}
    if SEEN and (d or {}).get("served") == "legacy":
        before_pages = V.verify(q, SynthesisResult(SEEN["answer"], SEEN["claimed"], "pf7"),
                                [EvidenceResult(q, "policy", SEEN["items"], None, True, None, False, None)], []).pages
        before = presentation.source_chips(SimpleNamespace(items=SEEN["items"], pages=before_pages, answer=SEEN["answer"], best_section=""))
        after = presentation.source_chips(res)
        sup_before = {short(t) for t in supported_docs(SEEN["answer"], SEEN["items"], before_pages)}
        sup_after = {short(t) for t in supported_docs(res.answer, res.items, list(res.pages))}
        rec.update(before_chips=before, after_chips=after, before_pages=before_pages, after_pages=list(res.pages),
                   unsupported_doc_chips_before=[c for c in doc_chips(before) if c not in sup_before],
                   unsupported_doc_chips_after=[c for c in doc_chips(after) if c not in sup_after],
                   every_displayed_doc_supports_a_claim=all(c in sup_after for c in doc_chips(after)))
    else:
        rec.update(after_chips=presentation.source_chips(res), note="not a legacy model answer (Plan F, extractor or refusal)")
    out.append(rec)
    print("\n=== " + q)
    print("served:", rec["served"], "| answer:", rec["answer"][:160].replace("\n", " / "))
    if "before_chips" in rec:
        print("before:", rec["before_chips"])
        print("after: ", rec["after_chips"])
        print("unsupported before:", rec["unsupported_doc_chips_before"], "| unsupported after:", rec["unsupported_doc_chips_after"])
    else:
        print("chips:", rec["after_chips"], "|", rec["note"])
json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
