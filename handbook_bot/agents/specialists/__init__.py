"""
Plan F specialist framework: contracts, base class and registry.

Plan F is a findings-based two-layer architecture. A Coordinator decides which
domain specialists a question needs; each specialist reasons over its own
scoped evidence and returns structured findings; the deterministic Python
orchestrator combines them, runs the independent Verifier and returns the
answer. This package holds only the foundation of that design:

    contracts   the typed shapes exchanged between Coordinator, specialists,
                orchestrator and Verifier
    base        the interface every specialist implements
    registry    resolves a canonical specialist id to an implementation
    scoping     shared source scoping: owned sections to page ranges, the
                section guard for shared pages
    evidence    shared quote units, quote quality and quote selection
    domain_cues the shared Research ownership lexicon read by the
                Coordinator, the Research and the Teaching specialists
    details     shared requested-detail primitives (values, money, timing,
                limits, authority)
    teaching    the Teaching & Learning Specialist, deterministic version 1:
                source-scoped retrieval, extractive findings, no LLM call
    research    the Research & Innovation Specialist, deterministic version 1,
                built on the shared helpers, never on the Teaching module
    faculty_services
                the Faculty Services & HR Specialist, deterministic version 1,
                built on the shared helpers and the shared HR cue lexicon
    dispatch    the seam from a Coordinator decision to specialist findings:
                runs the executable specialists, keeps the rest pending

Teaching, Research and Faculty Services & HR are implemented; no Coordinator or handoff execution
lives here, and nothing in the current chatbot runtime imports this package.
It becomes active only when a later phase wires it into
``handbook_bot.orchestrator``.

This file deliberately imports nothing, so importing one module of the package
never drags in another.
"""
