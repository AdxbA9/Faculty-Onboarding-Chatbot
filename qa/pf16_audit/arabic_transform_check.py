"""PF-16: does the deterministic Arabic -> English query keep what the question means? (no knowledge base, no model)

    python -X utf8 qa/pf16_audit/arabic_transform_check.py <holdout.jsonl> <out.json> [--label before|after]

For each case: the English query, coverage, unmapped words, whether every ``must_keep`` term survives in the query, and
(after the PF-16 change) whether the meaning-loss guard flagged the rewrite as unsafe so that the validated model
translation would replace it when a client is configured.
"""
import json
import sys

sys.path.insert(0, ".")
from handbook_bot import arabic_query as Q  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else "run"
rows = []
for line in open(src, encoding="utf-8"):
    if not line.strip():
        continue
    c = json.loads(line)
    t = Q.transform_query(c["question"])
    q = (t.english_query or "").lower()
    kept = all(k.lower() in q for k in c.get("must_keep") or [])
    flags = list(getattr(t, "meaning_loss", None) or [])
    rows.append({"id": c["id"], "english_query": t.english_query, "coverage": round(t.coverage, 3), "unmapped": t.unmapped,
                 "must_keep": c.get("must_keep"), "meaning_kept": kept, "guard_flags": flags,
                 "would_use_model_translation": bool(flags) or t.coverage < 0.6})
    print("%-11s kept=%-5s cov=%.2f flags=%-28s %s" % (c["id"], kept, t.coverage, ",".join(flags) or "-", t.english_query))
n = len(rows)
summary = {"label": label, "cases": n, "meaning_kept": [sum(1 for r in rows if r["meaning_kept"]), n],
           "lost_and_flagged": [sum(1 for r in rows if not r["meaning_kept"] and r["would_use_model_translation"]), sum(1 for r in rows if not r["meaning_kept"])],
           "kept_but_flagged": [sum(1 for r in rows if r["meaning_kept"] and r["guard_flags"]), sum(1 for r in rows if r["meaning_kept"])]}
json.dump({"summary": summary, "rows": rows}, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(summary))
