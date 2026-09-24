"""Plan F foundation: import hygiene in fresh interpreters, no import cycles,
inactive budgets, and existing consumers unaffected."""
from __future__ import annotations

import ast
import dataclasses
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from handbook_bot import config
from handbook_bot.qa import QAResult

REPO = Path(__file__).resolve().parents[1]
SPECIALISTS = REPO / "handbook_bot" / "agents" / "specialists"

NEW_MODULES = [
    "handbook_bot.agents.specialists",
    "handbook_bot.agents.specialists.contracts",
    "handbook_bot.agents.specialists.base",
    "handbook_bot.agents.specialists.registry",
]
EXISTING_MODULES = [
    "handbook_bot",
    "handbook_bot.config",
    "handbook_bot.agents.types",
    "handbook_bot.agents.router",
    "handbook_bot.agents.synthesis",
    "handbook_bot.agents.verifier",
    "handbook_bot.orchestrator",
    "handbook_bot.qa",
    "handbook_bot.retrieval",
    "ui.pipeline",
]


def fresh_python(code: str, env=None) -> subprocess.CompletedProcess:
    """Run ``code`` in a brand-new interpreter from the repository root."""
    merged = dict(os.environ)
    merged.setdefault("PYTHONIOENCODING", "utf-8")
    if env:
        merged.update(env)
    return subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=str(REPO), env=merged,
                          capture_output=True, text=True, timeout=180)


def assert_ok(proc: subprocess.CompletedProcess) -> str:
    assert proc.returncode == 0, "fresh interpreter failed:\n" + proc.stderr
    return proc.stdout


# ---------------------------------------------------------------------------
# Fresh-interpreter imports
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("module", NEW_MODULES)
def test_each_new_module_imports_alone_in_a_fresh_interpreter(module):
    out = assert_ok(fresh_python(
        "import importlib, sys\n"
        f"importlib.import_module({module!r})\n"
        "print('orchestrator_loaded', 'handbook_bot.orchestrator' in sys.modules)\n"
    ))
    # The foundation must never pull the runtime in behind it.
    assert "orchestrator_loaded False" in out.strip().splitlines()[-1]


def test_existing_modules_still_import_in_a_fresh_interpreter():
    assert_ok(fresh_python(
        "import importlib\n"
        + "".join(f"importlib.import_module({m!r})\n" for m in EXISTING_MODULES)
        + "print('ok')\n"
    ))


def test_runtime_first_then_foundation_has_no_cycle():
    assert_ok(fresh_python(
        "import handbook_bot.orchestrator\n"
        "import handbook_bot.agents.verifier\n"
        "from handbook_bot.agents.specialists.registry import SpecialistRegistry\n"
        "from handbook_bot.agents.specialists.contracts import SpecialistTask\n"
        "SpecialistRegistry(); SpecialistTask('t', 'q')\n"
        "print('ok')\n"
    ))


def test_ui_chat_imports_when_nicegui_is_available():
    if importlib.util.find_spec("nicegui") is None:
        pytest.skip("nicegui not installed in this environment")
    assert_ok(fresh_python("import ui.chat\nprint('ok')\n"))


def test_evaluation_scripts_import_in_a_fresh_interpreter():
    assert_ok(fresh_python(
        "import importlib.util, pathlib\n"
        "for name in ('run_eval', 'score', 'compare_runs'):\n"
        "    path = pathlib.Path('eval') / (name + '.py')\n"
        "    spec = importlib.util.spec_from_file_location(name, path)\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(module)\n"
        "print('ok')\n"
    ))


# ---------------------------------------------------------------------------
# Static cycle guard: what the foundation is allowed to import
# ---------------------------------------------------------------------------
def _import_roots(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add(("." * node.level) + (node.module or ""))
    return roots


def test_contracts_module_imports_only_the_standard_library():
    assert _import_roots(SPECIALISTS / "contracts.py") <= {"__future__", "dataclasses", "enum", "typing"}


def test_base_and_registry_import_only_the_foundation():
    assert _import_roots(SPECIALISTS / "base.py") <= {"__future__", "abc", "typing", ".contracts"}
    assert _import_roots(SPECIALISTS / "registry.py") <= {"__future__", "typing", ".base", ".contracts"}
    assert _import_roots(SPECIALISTS / "__init__.py") == set()


# ---------------------------------------------------------------------------
# Budgets: declared, namespaced, inactive
# ---------------------------------------------------------------------------
def test_plan_f_budgets_are_declared_with_the_agreed_values():
    assert config.PLAN_F_ENABLED is False
    assert config.PLAN_F_MAX_SPECIALISTS == 3
    assert config.PLAN_F_MAX_SUBTASKS == 3
    assert config.PLAN_F_MAX_HANDOFF_DEPTH == 1
    assert config.PLAN_F_MAX_RETRY == 1
    assert config.PLAN_F_MAX_LLM_CALLS == 5
    assert config.PLAN_F_MAX_AGENT_CALLS == 4


def test_milestone_1_budgets_are_unchanged():
    assert config.MAX_LLM_CALLS == 2
    assert config.MAX_VERIFY_RETRIES == 1
    assert config.PLANNER_ENABLED is False


def test_plan_f_budgets_are_environment_overridable_like_the_rest_of_config():
    out = assert_ok(fresh_python(
        "from handbook_bot import config\n"
        "print(config.PLAN_F_MAX_LLM_CALLS, config.PLAN_F_ENABLED, config.MAX_LLM_CALLS)\n",
        env={"PLAN_F_MAX_LLM_CALLS": "7", "PLAN_F_ENABLED": "1"},
    ))
    # Only the last line is ours: PyMuPDF may print a deprecation notice on import.
    assert out.strip().splitlines()[-1].split() == ["7", "True", "2"]   # Milestone 1 budget unaffected


def test_runtime_modules_do_not_reference_plan_f_yet():
    runtime = ["orchestrator.py", "qa.py", "retrieval.py", "agents/router.py",
               "agents/synthesis.py", "agents/verifier.py"]
    for name in runtime:
        text = (REPO / "handbook_bot" / name).read_text(encoding="utf-8")
        assert "PLAN_F" not in text and "specialists" not in text, name


# ---------------------------------------------------------------------------
# Existing result contract untouched
# ---------------------------------------------------------------------------
def test_qaresult_fields_are_unchanged_by_this_phase():
    assert [f.name for f in dataclasses.fields(QAResult)] == [
        "answer", "pages", "best_section", "evidence", "query_type", "items", "used_llm",
        "timings", "num_candidates", "num_reranked",
        "agent_trace", "llm_calls", "retried", "sub_questions",
    ]
