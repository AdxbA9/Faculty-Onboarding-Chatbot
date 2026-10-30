"""Print, for every Plan F answer served in the ACTIVE run of i4_results.json, the exact source chunk (production chunking of the
real PDF, no models) and the surrounding printed page text, so each accepted answer can be reviewed against the handbook.
Usage: python -X utf8 qa/plan_f_i4/review_accepted.py . <i4_results.json>"""
import json
import os
import sys

REPO, RESULTS = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))
os.chdir(REPO)
import institutional_support as IS                                   # noqa: E402
from handbook_bot.chunking import build_chunks                       # noqa: E402

pages = list(IS.handbook_pages())
chunks, metadata = build_chunks(pages)
R = json.load(open(RESULTS, encoding="utf-8"))
direct = {r["question"]: r for r in R["direct"]}
accepted = [r for r in R["results"] if r["mode"] == "active" and (r.get("diagnostics") or {}).get("served") == "plan_f"]
print("accepted answers:", len(accepted))
for n, r in enumerate(accepted, 1):
    dg = r["diagnostics"]
    v = (direct.get(r["question"]) or {}).get("direct") or {}
    print("\n" + "=" * 110)
    print("[%d] [%s] %s" % (n, r["domain"], r["question"]))
    print("    specialist=%s pages=%s section=%r outcome=%s | verifier: kind=%s completeness=%s dropped=%s" % (
        dg.get("specialist_id"), r["pages"], r["best_section"], dg.get("outcome"), v.get("request_kind"), v.get("completeness"), v.get("dropped")))
    print("    ANSWER: %r" % r["answer"])
    for it in r["items"]:
        idx, page, quote = it["idx"], it["page"], it["chunk"]
        src = chunks[idx] if isinstance(idx, int) and 0 <= idx < len(chunks) else None
        meta = metadata[idx] if src is not None else {}
        ok = src is not None and quote in src and meta.get("page") == page
        print("    item idx=%s page=%s chunk_type=%s quote-in-production-chunk=%s page-match=%s" % (idx, page, meta.get("chunk_type"), ok, meta.get("page") == page))
        if src is not None:
            pos = src.find(quote)
            before, after = src[max(0, pos - 160):pos], src[pos + len(quote):pos + len(quote) + 160]
            print("    context: ...%s[[%s]]%s..." % (before.replace("\n", " "), quote[:200].replace("\n", " "), after.replace("\n", " ")))
        page_text = IS.page_text(page) if page else ""
        print("    printed-page contains quote (whitespace-normalised): %s" % (IS.normalise(quote) in page_text))
