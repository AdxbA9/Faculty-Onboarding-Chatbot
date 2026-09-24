"""
Agents of the UoS Faculty Onboarding Chatbot (Senior Project Part 2).

A component is an *agent* only if it owns a decision. Everything else
(retrieval, reranking, chunking, the contact/count/date extractors) stays a
plain tool that an agent or the orchestrator calls.

    router      owns: which route handles the question, and whether that is ambiguous
    synthesis   owns: the wording of a grounded answer built from supplied evidence
    verifier    owns: whether an answer is supported, and which pages it may cite
    planner     (Milestone 2) owns: how a multi-part question is split
    evidence    (Milestone 2) owns: whether retrieved evidence is sufficient

    specialists/  (Plan F) the contracts, base class and registry for the four
                  domain specialists: teaching, research, faculty_services,
                  institutional. Foundation only; not wired into the runtime yet.

Agents never call each other. ``handbook_bot.orchestrator`` calls an agent,
receives a structured result from ``agents.types`` and decides what runs next.

This file deliberately imports nothing, so importing one agent can never drag
in another (or create an import cycle).
"""
