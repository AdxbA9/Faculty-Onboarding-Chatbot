# D-111 in brief (post-freeze PF-3C)

The full record is D-111 in `docs/DECISIONS.md`. This page lists what a reviewer needs to check a served guide answer.

## One path for guide evidence

- The guide-enabled registry (`PLAN_F_GUIDE_SCOPES` on) wraps the same I-9 specialist objects. The frozen extraction
  never reads a guide; handbook findings are verified exactly as at I-9.
- A clause is answered from a guide only when it names the system of a guide approved for the owning specialist (D-110)
  and asks a system question. Otherwise, or when the guide path finds no provably complete, fully covered evidence, the
  frozen specialist's I-9 result for that clause stands.

## What metadata may bind, and what text must state

| Bound by the registry (need not be printed in the evidence) | Must be printed in the evidence |
|---|---|
| the guide's declared system names (`source_scopes.SYSTEM_NAMES`) | every other content word of the clause: action, object, conditions, numbers |
| the first-person or faculty actor, only in a faculty or instructor guide (`audience`) | every other role (Dean, Chair, student) and every remaining named entity |

## Which asks a guide may answer

| Answered from a guide | Never answered from a guide |
|---|---|
| procedure, location, contact, value, yes/no, definition, enumeration, selection | policy, entitlement, obligation, deadline or timing, authority or approval power, workload and other handbook subjects |

## What counts as a complete procedure

- every step of ONE procedure, in source order, each an exact quote of its own chunk;
- one ordered list shape (bullets; 1..n with lettered sub-items; A..n), at least two numbered steps when numbered;
- a real heading (a heading row or a lead-in row, not carried, not a sentence fragment) that names the asked action or
  object; a heading that names another action is refused ("transfer/send" and "access/log in/sign in" are the only
  synonyms, as the guides use them);
- no continuation onto the next page under no heading and no start on the previous one; a numbered list continued at the
  top of the next page is merged and cited on both pages;
- a step cut by a continuation row is completed only from the page's own prose that holds the whole step, else refused;
- no step under a registered source conflict.

One item of an unordered list may answer alone only when it alone states the whole ask.

## Rejections

| Problem | Outcome |
|---|---|
| source not approved, wrong audience, system not named, policy or authority ask | `PLAN_F_REJECTED_BY_SOURCE_AUTHORITY` |
| a term or entity not stated, an incomplete or reordered procedure, a forged record | `PLAN_F_REJECTED_BY_SUBJECT_GUARD` |
| a registered conflict (the iThenticate yearly quota) | `PLAN_F_REJECTED_BY_SOURCE_CONFLICT` |
| guide evidence without the D-111 record | `PLAN_F_REJECTED_BY_SOURCE_AUTHORITY` |
