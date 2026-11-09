"""PF-16 database reliability probe on SYNTHETIC databases only (never var/app.db).

    python -X utf8 qa/pf16_audit/db_reliability.py <repo> <out dir>

1. round trip: two guests, English and Arabic turns (Arabic answer, English intermediate, translation state, evidence),
   progress, positions, preferences and display names; a NEW Store object on the same file (what a restart does) must
   return byte-identical content, and the other guest must see none of it;
2. backup and recovery with the real CLI (``python -m persistence.manage``): check -> backup -> damage the live file ->
   check reports the damage -> restore -> check -> content identical;
3. a writer process killed in the middle of writing (TerminateProcess / SIGKILL): the file opens, integrity ok, no partial
   assistant turn (every assistant message has its answer row);
4. concurrency: 8 guests on 8 threads, 40 operations each: no error, no cross-owner row;
5. latency of the store operations the pages use (median and 95th percentile over repeated calls).
Writes ``db_reliability.json``.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import threading
import time

REPO, OUT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
sys.path.insert(0, REPO)
os.chdir(REPO)
WORK = os.path.join(OUT, "db_work")
if os.path.isdir(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK)

from handbook_bot.qa import QAResult            # noqa: E402
from persistence import Store, db               # noqa: E402
from persistence.store import AccessError       # noqa: E402

report = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}
AR_Q = "ما هو رقم هاتف قسم الموارد البشرية؟"   # Arabic HR phone question
AR_A = "رقم الهاتف هو +971 6 5050023."
ITEM = {"chunk": "Human Resources +971 6 5050023 +971 6 5585200", "meta": {"page": 54, "source_id": "uos_faculty_handbook_2025_26",
        "source_title": "UOS Faculty Handbook 2025/26", "section": "Contact Information", "chunk_id": 101}, "score": 3.2}


def snapshot(store, owner, cid):
    turns = store.load_messages(owner, cid)
    out = []
    for t in turns:
        r = t.get("result")
        out.append({"role": t["role"], "content": t["content"], "answer": getattr(r, "answer", None), "original": getattr(r, "original_answer", None),
                    "lang": getattr(r, "answer_language", None), "pages": getattr(r, "pages", None), "translation": getattr(r, "translation", None),
                    "items": [(i.get("chunk"), (i.get("meta") or {}).get("page")) for i in (getattr(r, "items", None) or [])], "stored": t.get("stored")})
    return out


# ---- 1. round trip -------------------------------------------------------------------------------------------------------
path = os.path.join(WORK, "roundtrip.db")
s1 = Store(path)
a, b = s1.owner_for_token("guest-a-token"), s1.owner_for_token("guest-b-token")
conv_en = s1.create_conversation(a, "HR phone", "en")["id"]
s1.append_user_message(a, conv_en, "What is the phone number of the Human Resources Department?", client_token="t1")
s1.append_assistant_message(a, conv_en, QAResult(answer="The phone number is +971 6 5050023.", pages=[54], items=[ITEM]), {"served": "plan_f", "routing_final": ["institutional"]}, 812.0, client_token="t1a", corpus_key="c8987b6079b11f39")
conv_ar = s1.create_conversation(a, AR_Q, "ar")["id"]
s1.append_user_message(a, conv_ar, AR_Q, client_token="t2", answer_language="ar")
s1.append_assistant_message(a, conv_ar, QAResult(answer=AR_A, pages=[54], items=[ITEM], answer_language="ar", original_answer="The phone number is +971 6 5050023.",
                                                 translation={"status": "translated", "llm_calls": 1, "check": {"ok": True}}), {"served": "plan_f"}, 1500.0, client_token="t2a", corpus_key="c8987b6079b11f39")
s1.set_preferences(a, interface_lang="ar", theme="dark", text_size="large", reduced_motion="on", display_name="د. سارة")
s1.set_procedure_position(a, "bb_announcement", 2)
before = {"en": snapshot(s1, a, conv_en), "ar": snapshot(s1, a, conv_ar), "prefs": s1.get_preferences(a), "pos": s1.procedure_positions(a)}
s2 = Store(path)                                                     # a new object on the same file: what a restart does
after = {"en": snapshot(s2, a, conv_en), "ar": snapshot(s2, a, conv_ar), "prefs": s2.get_preferences(a), "pos": s2.procedure_positions(a)}
foreign = {}
for name, fn in (("load", lambda: s2.load_messages(b, conv_ar)), ("rename", lambda: s2.rename_conversation(b, conv_ar, "x")),
                 ("append", lambda: s2.append_user_message(b, conv_ar, "x"))):
    try:
        fn()
        foreign[name] = "ALLOWED"
    except AccessError:
        foreign[name] = "refused"
foreign["delete"] = "refused" if s2.delete_conversation(b, conv_ar) is False else "ALLOWED"
foreign["get"] = "refused" if s2.get_conversation(b, conv_ar) is None else "ALLOWED"
foreign["list_b"] = len(s2.list_conversations(b))
report["roundtrip"] = {"identical_after_reopen": before == after, "arabic_question_bytes_equal": after["ar"][0]["content"] == AR_Q,
                       "arabic_answer_equal": after["ar"][1]["answer"] == AR_A, "english_intermediate_kept": after["ar"][1]["original"],
                       "translation_state_kept": after["ar"][1]["translation"], "citations_kept": after["ar"][1]["pages"], "evidence_items": after["ar"][1]["items"],
                       "saved_flag_on_replay": [t["stored"] for t in after["ar"]], "prefs": after["prefs"], "positions": after["pos"], "foreign_access_by_b": foreign}

# ---- 2. backup and recovery through the CLI ------------------------------------------------------------------------------------
def cli(*args):
    p = subprocess.run([sys.executable, "-X", "utf8", "-m", "persistence.manage", "--db", path, *args], capture_output=True, text=True, encoding="utf-8", cwd=REPO)
    return p.returncode, (p.stdout or "") + (p.stderr or "")

rc_check, out_check = cli("check")
backup_file = os.path.join(WORK, "roundtrip.backup.db")
rc_backup, out_backup = cli("backup", backup_file)
for suffix in ("-wal", "-shm"):
    if os.path.exists(path + suffix):
        os.remove(path + suffix)
with open(path, "r+b") as f:                                          # damage: overwrite part of the first page after the header
    f.seek(100)
    f.write(b"\x00\xff" * 400)
rc_damaged, out_damaged = cli("check")
rc_restore, out_restore = cli("restore", backup_file, "--yes")
rc_after, out_after = cli("check")
s3 = Store(path)
restored_equal = {"en": snapshot(s3, a, conv_en), "ar": snapshot(s3, a, conv_ar), "prefs": s3.get_preferences(a), "pos": s3.procedure_positions(a)} == before
report["backup_recovery"] = {"check_before": [rc_check, out_check[-300:]], "backup": [rc_backup, out_backup[-300:]], "check_after_damage": [rc_damaged, out_damaged[-400:]],
                             "restore": [rc_restore, out_restore[-300:]], "check_after_restore": [rc_after, out_after[-300:]], "content_identical_after_restore": restored_equal,
                             "kept_damaged_copy": sorted(n for n in os.listdir(WORK) if "before-restore" in n)}

# ---- 3. a writer killed mid-write ----------------------------------------------------------------------------------------------
kpath = os.path.join(WORK, "killed.db")
writer = r'''
import sys, time
sys.path.insert(0, %r)
from persistence import Store
from handbook_bot.qa import QAResult
s = Store(%r); o = s.owner_for_token("killer")
i = 0
while True:
    c = s.create_conversation(o, "c%%d" %% i)["id"]
    s.append_user_message(o, c, "question %%d" %% i, client_token="u%%d" %% i)
    s.append_assistant_message(o, c, QAResult(answer="answer %%d" %% i, pages=[1], items=[{"chunk": "x" * 2000, "meta": {"page": 1, "source_id": "s"}}]), {"served": "legacy"}, 1.0, client_token="a%%d" %% i)
    i += 1
''' % (REPO, kpath)
proc = subprocess.Popen([sys.executable, "-c", writer], cwd=REPO)
time.sleep(4.0)
proc.kill()
proc.wait()
conn = sqlite3.connect(kpath)
integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
fk = conn.execute("PRAGMA foreign_key_check").fetchall()
n_conv = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
n_user = conn.execute("SELECT COUNT(*) FROM messages WHERE role='user'").fetchone()[0]
n_asst = conn.execute("SELECT COUNT(*) FROM messages WHERE role='assistant'").fetchone()[0]
orphans = conn.execute("SELECT COUNT(*) FROM messages m WHERE role='assistant' AND NOT EXISTS (SELECT 1 FROM answers x WHERE x.message_id = m.id)").fetchone()[0]
conn.close()
report["killed_writer"] = {"integrity": integrity, "foreign_key_violations": len(fk), "conversations": n_conv, "user_messages": n_user,
                           "assistant_messages": n_asst, "assistant_without_answer_row": orphans,
                           "store_reopens": bool(Store(kpath).list_conversations(Store(kpath).owner_for_token("killer"), limit=5) is not None)}

# ---- 4. concurrency ----------------------------------------------------------------------------------------------------------------
cpath = os.path.join(WORK, "concurrent.db")
cs = Store(cpath)
errors, owners = [], {}


def guest(n):
    try:
        st = Store(cpath)
        o = st.owner_for_token("g%d" % n)
        owners[n] = o
        for i in range(10):
            c = st.create_conversation(o, "g%d-%d" % (n, i))["id"]
            st.append_user_message(o, c, "q%d" % i, client_token="q%d-%d" % (n, i))
            st.set_preferences(o, theme="dark" if i % 2 else "light")
            st.list_conversations(o)
    except Exception as exc:                                                    # recorded, not raised
        errors.append("%s: %s" % (type(exc).__name__, exc))


threads = [threading.Thread(target=guest, args=(n,)) for n in range(8)]
t0 = time.perf_counter()
[t.start() for t in threads]
[t.join() for t in threads]
conc_s = time.perf_counter() - t0
cross = sum(1 for n, o in owners.items() for c in cs.list_conversations(o) if not c["title"].startswith("g%d-" % n))
report["concurrency"] = {"guests": 8, "operations_per_guest": 40, "errors": errors, "per_guest_conversations": sorted(len(cs.list_conversations(o)) for o in owners.values()),
                         "cross_owner_rows": cross, "seconds": round(conc_s, 2)}

# ---- 5. latency ----------------------------------------------------------------------------------------------------------------------
lpath = os.path.join(WORK, "latency.db")
ls = Store(lpath)
lo = ls.owner_for_token("lat")


def timed(fn, n):
    xs = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t) * 1000)
    xs.sort()
    return {"n": n, "median_ms": round(statistics.median(xs), 2), "p95_ms": round(xs[int(0.95 * (n - 1))], 2), "max_ms": round(xs[-1], 2)}


convs = [ls.create_conversation(lo, "c%d" % i)["id"] for i in range(50)]
k = {"i": 0}


def write_turn():
    c = convs[k["i"] % 50]
    k["i"] += 1
    ls.append_user_message(lo, c, "question %d" % k["i"], client_token="lu%d" % k["i"])
    ls.append_assistant_message(lo, c, QAResult(answer="answer", pages=[54], items=[ITEM]), {"served": "plan_f"}, 1.0, client_token="la%d" % k["i"])


report["latency"] = {"write_question_and_answer": timed(write_turn, 200), "list_conversations_50": timed(lambda: ls.list_conversations(lo), 100),
                     "load_conversation_8_turns": timed(lambda: ls.load_messages(lo, convs[0]), 100),
                     "set_preferences": timed(lambda: ls.set_preferences(lo, theme="dark"), 100),
                     "get_preferences": timed(lambda: ls.get_preferences(lo), 100)}
report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
with open(os.path.join(OUT, "db_reliability.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1, default=str)
print(json.dumps(report, ensure_ascii=False, default=str, indent=1)[:6000])
