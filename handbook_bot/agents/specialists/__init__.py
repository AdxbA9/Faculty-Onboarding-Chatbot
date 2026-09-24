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

No specialist, Coordinator or handoff logic lives here yet, and nothing in the
current chatbot runtime imports this package. It becomes active only when a
later phase wires it into ``handbook_bot.orchestrator``.

This file deliberately imports nothing, so importing one module of the package
never drags in another.
"""
