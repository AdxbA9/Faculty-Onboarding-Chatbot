"""
The onboarding pages (post-freeze PF-14, D-125): the checklist dashboard (``/onboarding``), the guided procedure
navigator (``/onboarding/procedure/{procedure_id}``) and the services and documents finder (``/resources``).

What this module decides: layout and wording of the three pages (``ui.i18n`` in English and Arabic), which catalog
entries are shown and how, and when a progress row is written (only on an explicit control the user operated). What it
never decides: the answer to any question. "Ask the assistant" and "Ask about this step" open the chat page with the
question pre-filled in the composer and the task as context; the unchanged controlled pipeline answers there, and the
answer never changes a progress row.

Identity and preferences are the PF-13 ones: the browser's signed session cookie, resolved by ``chat.page_context``;
the owner row is created by the first write (a progress change, a theme or language choice), never by a visit.
"""
from __future__ import annotations

import html
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlencode

from fastapi import Request
from nicegui import run, ui

from onboarding import catalog as C
from onboarding.finder import search
from onboarding.links import anchor, safe_url
from onboarding.progress import dashboard, status_of
from persistence import StoreError

from . import chat, i18n
from .i18n import t
from .styles import GLOBAL_CSS, ICONS

KIND_KEYS = {"required": "kind_required", "recommended": "kind_recommended", "optional": "kind_optional", "info": "kind_info"}
STATUS_KEYS = {"not_started": "status_not_started", "in_progress": "status_in_progress", "completed": "status_completed", "not_applicable": "status_not_applicable"}
CATEGORY_ICONS = {"teaching": "teaching", "research": "research", "faculty_services": "services", "institutional": "institutional"}
TYPE_KEYS = {"document": "rf_type_document", "department": "rf_type_department", "website": "rf_type_website"}


def esc(text: Any, quote_: bool = False) -> str:
    return html.escape(str(text if text is not None else ""), quote=quote_)


def title_of(obj: Any, lang: str) -> str:
    return getattr(obj, "title_ar" if lang == "ar" else "title_en", "") or getattr(obj, "title_en", "")


def text_of(obj: Any, field: str, lang: str) -> str:
    return getattr(obj, "%s_%s" % (field, "ar" if lang == "ar" else "en"), "") or getattr(obj, field + "_en", "")


def source_label(source_id: str) -> str:
    return C.SOURCE_TITLES.get(source_id, source_id)


def task_question(task: C.Task, lang: str) -> str:
    """The pre-filled question about a task. The task is named by its English title in both languages: the documents are
    English, and the PF-11 layer keeps Latin text as it is while it transforms the Arabic frame, so the Arabic question
    retrieves the same evidence as the English one (an Arabic title would be re-translated word by word)."""
    return t(lang, "ob_task_question", task=task.title_en)


def ask_url(task_id: Optional[str], question: str) -> str:
    params = {"ask": question}
    if task_id:
        params["task"] = task_id
    return "/?" + urlencode(params, quote_via=quote)


# =====================================================================================================================
# Shared shell
# =====================================================================================================================
class Page:
    """One page build: context, shell and the small helpers every page needs."""

    def __init__(self, request: Request, active: str) -> None:
        self.ctx = chat.page_context(request)
        self.lang = self.ctx["lang"]
        self.active = active
        self.refs: Dict[str, Any] = {}

    def L(self) -> str:
        return self.lang

    def store(self):
        return self.ctx["store"]

    def owner_id(self) -> Optional[int]:
        return self.ctx.get("owner_id")

    def ensure_owner(self) -> Optional[int]:
        """The owner id, creating the row on this first write (never on a visit)."""
        if self.ctx.get("owner_id") is not None:
            return self.ctx["owner_id"]
        store, token = self.ctx.get("store"), self.ctx.get("owner_token")
        if store is None or not token:
            return None
        try:
            self.ctx["owner_id"] = store.owner_for_token(token)
        except StoreError as exc:
            chat._log.error("creating the owner failed: %s", type(exc).__name__)
            return None
        return self.ctx["owner_id"]

    def notify(self, text: str, kind: str = "info") -> None:
        el = self.refs.get("live")
        if el is not None:
            chat._safe(setattr, el, "content", '<div class="page-note %s" role="status">%s%s</div>' % (kind, ICONS["check" if kind == "ok" else "info"], esc(text)))

    async def save_pref(self, **changes: Any) -> None:
        owner = self.ensure_owner()
        store = self.store()
        if owner is None or store is None:
            return
        try:
            await run.io_bound(lambda: store.set_preferences(owner, **changes))
        except StoreError as exc:
            chat._log.error("saving preferences failed: %s", type(exc).__name__)

    def shell(self, page_title: str):
        lang = self.lang
        ui.add_head_html(GLOBAL_CSS)
        ui.query("html").props('data-theme=%s lang=%s dir=%s' % (self.ctx["theme"], lang, i18n.DIRECTION[lang]))
        root = ui.element("div").classes("app-root ob-root")
        with root:
            ui.html('<a class="skip-link" href="#ob-main">%s</a>' % esc(t(lang, "skip_to_content")))
            with ui.element("header").classes("ob-topbar").props('role="banner"'):
                ui.html('<a class="ob-brand" href="/"><img src="%s" alt="%s"/><span><span class="brand-org">%s</span><span class="brand-name">%s</span></span></a>'
                        % (chat._LOGO_URL, esc(t(lang, "logo_alt"), True), esc(t(lang, "org")), esc(t(lang, "app_name"))))
                with ui.element("nav").classes("ob-nav").props('aria-label="%s"' % esc(t(lang, "nav_aria"), True)):
                    for key, href, label in (("chat", "/", "nav_chat"), ("onboarding", "/onboarding", "nav_onboarding"), ("resources", "/resources", "nav_resources")):
                        current = ' aria-current="page"' if key == self.active else ""
                        ui.html('<a class="ob-nav-link%s" href="%s"%s>%s%s</a>' % (" active" if key == self.active else "", href, current, ICONS["arrow" if key == "chat" else ("check" if key == "onboarding" else "search")], esc(t(lang, label))))
                with ui.element("div").classes("ob-topbar-right"):
                    ui.button(t(lang, "lang_switch"), on_click=self._toggle_lang).props('flat dense no-caps aria-label="%s" lang="%s"' % (esc(t(lang, "lang_switch_aria"), True), "ar" if lang == "en" else "en")).classes("topbar-btn topbar-lang").mark("lang")
                    theme_btn = ui.button(on_click=self._toggle_theme).props('flat dense aria-label="%s"' % esc(t(lang, "theme_to_light" if self.ctx["theme"] == "dark" else "theme_to_dark"), True)).classes("topbar-btn").mark("theme")
                    with theme_btn:
                        ui.html(ICONS["moon"] + ICONS["sun"])
            main = ui.element("main").classes("ob-main").props('id="ob-main"')
            with main:
                self.refs["live"] = ui.html("").classes("ob-live").props('aria-live="polite"')
                if self.ctx.get("storage_notice"):
                    ui.html('<div class="banner warn" role="status">%s<div><strong>%s</strong> %s</div></div>' % (ICONS["info"], esc(t(lang, "storage_unavailable")), esc(t(lang, "ob_storage_unavailable"))))
        return main

    async def _toggle_lang(self) -> None:
        new = "ar" if self.lang == "en" else "en"
        await self.save_pref(interface_lang=new)
        ui.navigate.to(self._same_page_with(lang=new))

    async def _toggle_theme(self) -> None:
        new = "light" if self.ctx["theme"] == "dark" else "dark"
        self.ctx["theme"] = new
        chat._safe(ui.query("html").props, "data-theme=%s" % new)
        await self.save_pref(theme=new)

    def _same_page_with(self, **params: str) -> str:
        request: Request = self.ctx["request"]
        query = {k: v for k, v in request.query_params.items() if k != "lang"}
        query.update(params)
        return request.url.path + ("?" + urlencode(query, quote_via=quote) if query else "")


# =====================================================================================================================
# Dashboard
# =====================================================================================================================
def build_dashboard(request: Request) -> None:
    page = Page(request, "onboarding")
    lang = page.lang
    category = request.query_params.get("category") or "all"
    if category not in C.CATEGORIES:
        category = "all"
    highlight = request.query_params.get("task") or ""
    store, owner = page.store(), page.owner_id()
    progress: Dict[str, Dict[str, Any]] = {}
    if store is not None and owner is not None:
        try:
            progress = store.task_progress(owner)
        except StoreError as exc:
            chat._log.error("loading progress failed: %s", type(exc).__name__)
            page.ctx["storage_notice"] = "storage_unavailable"
    state = {"progress": progress, "category": category}
    main = page.shell(t(lang, "ob_title"))

    def status_label(status: str) -> str:
        return t(lang, STATUS_KEYS.get(status, "status_not_started"))

    def render_summary() -> None:
        container = page.refs.get("summary")
        if container is None:
            return
        container.clear()
        d = dashboard(state["progress"], C.tasks_by_category())
        with container:
            with ui.element("section").classes("ob-summary").props('aria-label="%s"' % esc(t(lang, "ob_progress"), True)):
                ui.html('<div class="ob-ring" role="img" aria-label="%s" style="--pct:%d"><span class="ob-ring-value">%d%%</span></div>'
                        % (esc(t(lang, "ob_percent", percent=d["percent"]), True), d["percent"], d["percent"]))
                with ui.element("div").classes("ob-summary-body"):
                    ui.html('<h2 class="ob-h2">%s</h2>' % esc(t(lang, "ob_progress")))
                    ui.html('<dl class="ob-stats">' + "".join('<div class="ob-stat"><dt>%s</dt><dd data-stat="%s">%d</dd></div>' % (esc(t(lang, key)), stat, d[stat]) for key, stat in (
                        ("ob_applicable", "applicable"), ("ob_completed", "completed"), ("ob_in_progress", "in_progress"), ("ob_remaining", "remaining"))) + "</dl>")
                    chips = []
                    for cat in C.CATEGORIES:
                        bc = d["by_category"][cat]
                        chips.append('<span class="chip cat">%s %s <b>%d/%d</b></span>' % (ICONS[CATEGORY_ICONS[cat]], esc(i18n.domain_name(lang, cat)), bc["completed"], bc["applicable"]))
                    ui.html('<div class="ob-cat-chips">%s</div>' % "".join(chips))
                    if d["applicable"] == 0:
                        ui.html('<p class="ob-note">%s</p>' % esc(t(lang, "ob_none_applicable")))
                    elif d["all_done"]:
                        ui.html('<p class="ob-note ok">%s%s</p>' % (ICONS["check"], esc(t(lang, "ob_all_done"))))
                    else:
                        items = "".join('<li><a href="/onboarding?task=%s#task-%s">%s</a> <span class="kind %s">%s</span></li>' % (
                            tid, tid, esc(title_of(C.TASKS[tid], lang)), C.TASKS[tid].kind, esc(t(lang, KIND_KEYS[C.TASKS[tid].kind]))) for tid in d["next"])
                        ui.html('<h3 class="ob-h3">%s</h3><ul class="ob-next">%s</ul>' % (esc(t(lang, "ob_next")), items))

    async def set_status(task_id: str, status: str) -> None:
        if status not in C.STATUSES or task_id not in C.TASKS:
            return
        owner = page.ensure_owner()
        st = page.store()
        if owner is None or st is None:
            page.notify(t(lang, "ob_save_failed"), "warn")
            return
        try:
            result = await run.io_bound(st.set_task_status, owner, task_id, status)
        except StoreError as exc:
            chat._log.error("saving progress failed: %s", type(exc).__name__)
            page.notify(t(lang, "ob_save_failed"), "warn")
            return
        state["progress"][task_id] = {"status": result["status"], "updated_at": result.get("updated_at"), "completed_at": None}
        if result["changed"]:
            page.notify(t(lang, "ob_marked", task=title_of(C.TASKS[task_id], lang), status=status_label(status)), "ok")
        render_summary()
        render_tasks()

    def render_tasks() -> None:
        container = page.refs.get("tasks")
        if container is None:
            return
        container.clear()
        tasks = C.tasks_by_category(None if state["category"] == "all" else state["category"])
        with container:
            if not tasks:
                ui.html('<div class="ob-empty">%s%s</div>' % (ICONS["info"], esc(t(lang, "ob_empty_filter"))))
                return
            for task in tasks:
                status = status_of(state["progress"], task.id)
                with ui.element("article").classes("ob-task" + (" done" if status == "completed" else "") + (" highlight" if task.id == highlight else "")) \
                        .props('id="task-%s" aria-labelledby="task-title-%s"' % (task.id, task.id)).mark("task"):
                    ui.html('<div class="ob-task-head"><span class="ob-task-cat">%s%s</span><span class="kind %s">%s</span>'
                            '<span class="ob-status-pill %s">%s</span></div>' % (
                                ICONS[CATEGORY_ICONS[task.category]], esc(i18n.domain_name(lang, task.category)), task.kind, esc(t(lang, KIND_KEYS[task.kind])),
                                status, esc(status_label(status))))
                    ui.html('<h3 class="ob-task-title" id="task-title-%s" dir="auto">%s</h3><p class="ob-task-desc" dir="auto">%s</p>' % (
                        task.id, esc(title_of(task, lang)), esc(text_of(task, "description", lang))))
                    if text_of(task, "timing", lang):
                        ui.html('<p class="ob-task-timing">%s<span>%s</span></p>' % (ICONS["clock"], esc(text_of(task, "timing", lang))))
                    evidence_rows = "".join('<li><span class="ob-ev-src">%s · %s%s</span><blockquote dir="ltr" lang="en">%s</blockquote></li>' % (
                        esc(source_label(e.source_id)), esc(t(lang, "ob_page", page=e.page)), (" · " + esc(e.section)) if e.section else "", esc(e.quote)) for e in task.evidence)
                    ui.html('<details class="ob-evidence"><summary>%s%s <span class="ob-ev-count">%d</span></summary><ul>%s</ul></details>' % (
                        ICONS["book"], esc(t(lang, "ob_evidence")), len(task.evidence), evidence_rows))
                    with ui.element("div").classes("ob-task-actions"):
                        if task.procedure_id:
                            ui.button(t(lang, "ob_open_guide"), on_click=lambda _e, tid=task.id, pid=task.procedure_id: ui.navigate.to("/onboarding/procedure/%s?task=%s&lang=%s" % (pid, tid, lang))) \
                                .props('unelevated no-caps dense icon=menu_book').classes("ob-btn primary").mark("open-guide")
                        ui.button(t(lang, "ob_ask"), on_click=lambda _e, tid=task.id: ui.navigate.to(ask_url(tid, task_question(C.TASKS[tid], lang)))) \
                            .props('outline no-caps dense icon=forum').classes("ob-btn").mark("ask-assistant")
                        ui.button(t(lang, "ob_resources"), on_click=lambda _e, tid=task.id: ui.navigate.to("/resources?task=%s&lang=%s" % (tid, lang))) \
                            .props('outline no-caps dense icon=search').classes("ob-btn").mark("find-resources")
                    with ui.element("div").classes("ob-task-status"):
                        ui.html('<span class="lbl">%s</span>' % esc(t(lang, "ob_status")))
                        options = {s: status_label(s) for s in C.STATUSES}
                        ui.select(options, value=status, on_change=lambda e, tid=task.id: set_status(tid, e.value)) \
                            .props('dense outlined options-dense aria-label="%s"' % esc(t(lang, "ob_status_aria", task=title_of(task, lang)), True)).classes("ob-status-select").mark("status-" + task.id)
                        if status != "completed":
                            ui.button(t(lang, "ob_mark_complete"), on_click=lambda _e, tid=task.id: set_status(tid, "completed")) \
                                .props('unelevated no-caps dense icon=task_alt').classes("ob-btn complete").mark("complete-" + task.id)

    def pick_category(cat: str) -> None:
        state["category"] = cat
        for key, btn in page.refs.get("filters", {}).items():
            chat._safe(btn.props, 'aria-pressed="%s"' % ("true" if key == cat else "false"))
            chat._safe(btn.classes, add="active") if key == cat else chat._safe(btn.classes, remove="active")
        render_tasks()

    with main:
        ui.html('<div class="ob-page-head"><h1 class="ob-h1">%s</h1><p class="ob-sub">%s</p></div>' % (esc(t(lang, "ob_title")), esc(t(lang, "ob_subtitle"))))
        page.refs["summary"] = ui.element("div").classes("ob-summary-wrap")
        with ui.element("div").classes("ob-filters").props('role="group" aria-label="%s"' % esc(t(lang, "ob_filter_aria"), True)):
            page.refs["filters"] = {}
            for cat in ("all",) + C.CATEGORIES:
                label = t(lang, "ob_filter_all") if cat == "all" else i18n.domain_name(lang, cat)
                btn = ui.button(label, on_click=lambda _e, c=cat: pick_category(c)).props('flat no-caps dense aria-pressed="%s"' % ("true" if cat == state["category"] else "false")) \
                    .classes("ob-filter" + (" active" if cat == state["category"] else "")).mark("filter-" + cat)
                page.refs["filters"][cat] = btn
        page.refs["tasks"] = ui.element("section").classes("ob-tasks").props('aria-label="%s"' % esc(t(lang, "ob_tasks_aria"), True))
        ui.html('<p class="ob-foot">%s</p>' % esc(t(lang, "ob_disclaimer")))
    render_summary()
    render_tasks()


# =====================================================================================================================
# Procedure navigator
# =====================================================================================================================
def build_procedure(request: Request, procedure_id: str) -> None:
    page = Page(request, "onboarding")
    lang = page.lang
    proc = C.procedure(procedure_id)
    task_id = request.query_params.get("task") or ""
    task = C.task(task_id)
    if task is not None and task.procedure_id != (proc.id if proc else None):
        task = None                                          # a forged pairing: the task is simply not shown as linked
    main = page.shell(t(lang, "pr_title"))
    if proc is None:
        with main:
            ui.html('<div class="ob-empty" role="alert">%s%s</div><p><a class="ob-link" href="/onboarding">%s</a></p>' % (ICONS["alert"], esc(t(lang, "pr_not_found")), esc(t(lang, "pr_back"))))
        return
    total = len(proc.steps)
    store, owner = page.store(), page.owner_id()
    start = 0
    if store is not None and owner is not None:
        try:
            start = min(store.procedure_position(owner, proc.id), total - 1)
        except StoreError:
            start = 0
    state = {"index": max(start, 0)}

    def step_question(step: C.Step) -> str:
        return t(lang, "pr_question", guide=proc.heading, step=step.text)

    def render_step() -> None:
        container = page.refs.get("step")
        if container is None:
            return
        container.clear()
        i = state["index"]
        step = proc.steps[i]
        with container:
            ui.html('<div class="pr-progress" role="progressbar" aria-valuemin="1" aria-valuemax="%d" aria-valuenow="%d" aria-label="%s"><span style="width:%d%%"></span></div>'
                    % (total, i + 1, esc(t(lang, "pr_step", n=i + 1, total=total), True), int(100 * (i + 1) / total)))
            flag = ""
            if step.flag == "conditional":
                flag = '<span class="pr-flag conditional">%s%s</span>' % (ICONS["info"], esc(t(lang, "pr_conditional")))
            elif step.flag == "after":
                flag = '<span class="pr-flag after">%s%s</span>' % (ICONS["clock"], esc(t(lang, "pr_after")))
            ui.html('<div class="pr-step-head"><span class="pr-counter">%s</span>%s</div>' % (esc(t(lang, "pr_step", n=i + 1, total=total)), flag))
            ui.html('<blockquote class="pr-step-text" dir="ltr" lang="en">%s</blockquote><p class="pr-quote-note">%s</p>' % (esc(step.text), esc(t(lang, "pr_quote_note"))))
            with ui.element("div").classes("pr-controls"):
                prev_btn = ui.button(t(lang, "pr_prev"), on_click=lambda: move(-1)).props('outline no-caps dense icon=chevron_left').classes("ob-btn").mark("prev")
                if i == 0:
                    prev_btn.props("disable")
                next_btn = ui.button(t(lang, "pr_next"), on_click=lambda: move(1)).props('unelevated no-caps dense icon-right=chevron_right').classes("ob-btn primary").mark("next")
                if i == total - 1:
                    next_btn.props("disable")
                ui.button(t(lang, "pr_ask_step"), on_click=lambda: ui.navigate.to(ask_url(task.id if task else None, step_question(step)))) \
                    .props('outline no-caps dense icon=forum').classes("ob-btn").mark("ask-step")
            ui.html('<details class="ob-evidence" open><summary>%s%s</summary><div class="pr-evidence"><span class="ob-ev-src">%s · %s · %s</span>'
                    '<a class="ob-link" href="/resources?task=%s&lang=%s">%s</a></div></details>' % (
                        ICONS["book"], esc(t(lang, "ob_evidence")), esc(source_label(proc.source_id)), esc(t(lang, "ob_page", page=step.page)),
                        esc(t(lang, "pr_chunk", chunk=step.chunk_id)), task_id if task else "", lang, esc(t(lang, "rf_open_in_finder"))))
            dots = "".join('<button type="button" class="pr-dot%s" aria-label="%s" data-step="%d"%s></button>' % (
                " current" if n == i else (" conditional" if s.flag == "conditional" else (" after" if s.flag == "after" else "")),
                esc(t(lang, "pr_step", n=n + 1, total=total), True), n, ' aria-current="step"' if n == i else "") for n, s in enumerate(proc.steps))
            ui.html('<div class="pr-dots" role="list">%s</div>' % dots).on("click", lambda e: jump(e), js_handler="(e) => { const b = e.target.closest('.pr-dot'); if (b) emit({step: Number(b.dataset.step)}); }")

    def jump(e) -> None:
        args = getattr(e, "args", None) or {}
        target = args.get("step") if isinstance(args, dict) else None
        if isinstance(target, int) and 0 <= target < total:
            state["index"] = target
            render_step()
            remember()

    def remember() -> None:
        """Keep the step the user moved to (Previous/Next/dots are interactions: the owner row is created by the first one;
        merely opening the page creates nothing)."""
        owner, st = page.ensure_owner(), page.store()
        if owner is None or st is None:
            return

        async def _save() -> None:
            try:
                await run.io_bound(st.set_procedure_position, owner, proc.id, state["index"])
            except StoreError as exc:
                chat._log.warning("saving the procedure position failed: %s", type(exc).__name__)
        chat.background_tasks.create(_save())

    def move(delta: int) -> None:
        state["index"] = min(max(state["index"] + delta, 0), total - 1)
        render_step()
        remember()

    async def mark_done() -> None:
        if task is None:
            return
        owner = page.ensure_owner()
        st = page.store()
        if owner is None or st is None:
            page.notify(t(lang, "ob_save_failed"), "warn")
            return
        try:
            result = await run.io_bound(st.set_task_status, owner, task.id, "completed")
        except StoreError as exc:
            chat._log.error("saving progress failed: %s", type(exc).__name__)
            page.notify(t(lang, "ob_save_failed"), "warn")
            return
        page.notify(t(lang, "ob_marked", task=title_of(task, lang), status=t(lang, "status_completed")), "ok")
        btn = page.refs.get("done")
        if btn is not None:
            chat._safe(btn.props, "disable")
            chat._safe(btn.set_text, t(lang, "pr_marked_done"))

    with main:
        crumbs = '<a class="ob-link" href="/onboarding?lang=%s">%s</a>' % (lang, esc(t(lang, "pr_back")))
        if task is not None:
            crumbs += ' <span class="sep">›</span> <a class="ob-link" href="/onboarding?task=%s&lang=%s#task-%s">%s</a>' % (task.id, lang, task.id, esc(title_of(task, lang)))
        ui.html('<nav class="ob-crumbs" aria-label="%s">%s</nav>' % (esc(t(lang, "pr_crumbs_aria"), True), crumbs))
        ui.html('<div class="ob-page-head"><h1 class="ob-h1" dir="auto">%s</h1><p class="ob-sub"><span class="chip page">%s</span> <span class="chip">%s</span> <span class="chip">%s</span></p>'
                '<p class="pr-heading">%s <q lang="en" dir="ltr">%s</q></p>%s</div>' % (
                    esc(title_of(proc, lang)), esc(source_label(proc.source_id)), esc(t(lang, "pr_pages", pages=", ".join(str(p) for p in proc.pages))),
                    esc(t(lang, "pr_version", date=proc.version)), esc(t(lang, "pr_source")), esc(proc.heading),
                    ('<p class="pr-conditions">%s<span>%s</span></p>' % (ICONS["info"], esc(text_of(proc, "conditions", lang)))) if text_of(proc, "conditions", lang) else ""))
        page.refs["step"] = ui.element("section").classes("pr-card").props('aria-label="%s"' % esc(t(lang, "pr_title"), True))
        if task is not None:
            with ui.element("section").classes("pr-task-box").props('aria-label="%s"' % esc(t(lang, "pr_task"), True)):
                ui.html('<h2 class="ob-h3">%s</h2><p dir="auto"><strong>%s</strong></p><p class="ob-note">%s</p>' % (esc(t(lang, "pr_task")), esc(title_of(task, lang)), esc(t(lang, "pr_done_note"))))
                status = "not_started"
                if store is not None and owner is not None:
                    try:
                        status = status_of(store.task_progress(owner), task.id)
                    except StoreError:
                        status = "not_started"
                page.refs["done"] = ui.button(t(lang, "pr_marked_done") if status == "completed" else t(lang, "pr_mark_done"), on_click=mark_done) \
                    .props('unelevated no-caps icon=task_alt' + (" disable" if status == "completed" else "")).classes("ob-btn complete").mark("mark-done")
    render_step()


# =====================================================================================================================
# Resource finder
# =====================================================================================================================
def build_finder(request: Request) -> None:
    page = Page(request, "resources")
    lang = page.lang
    task_id = request.query_params.get("task") or ""
    task = C.task(task_id)
    initial_query = (request.query_params.get("q") or "")[:200]
    category = request.query_params.get("category") or "all"
    if category not in C.CATEGORIES:
        category = "all"
    state = {"query": initial_query, "category": category, "task": task.id if task else None}
    main = page.shell(t(lang, "rf_title"))

    def render_results() -> None:
        container = page.refs.get("results")
        if container is None:
            return
        container.clear()
        results = search(state["query"], category=None if state["category"] == "all" else state["category"], task_id=state["task"], limit=40)
        count = page.refs.get("count")
        if count is not None:
            chat._safe(setattr, count, "content", esc(t(lang, "rf_results", n=len(results))))
        with container:
            if not results:
                with ui.element("div").classes("ob-empty").props('role="status"'):
                    ui.html('%s<span>%s</span>' % (ICONS["info"], esc(t(lang, "rf_no_results", query=state["query"]) if state["query"] else t(lang, "rf_no_results_filter"))))
                    if state["query"]:
                        ui.button(t(lang, "rf_ask_instead"), on_click=lambda: ui.navigate.to(ask_url(state["task"], state["query"]))).props('outline no-caps dense icon=forum').classes("ob-btn").mark("ask-instead")
                return
            for entry in results:
                res: C.Resource = entry["resource"]
                with ui.element("article").classes("rf-card" + (" contextual" if entry["contextual"] else "")).props('aria-labelledby="res-%s"' % res.id).mark("resource"):
                    ui.html('<div class="rf-head"><span class="rf-type %s">%s</span><span class="ob-task-cat">%s%s</span>%s</div>' % (
                        res.type, esc(t(lang, TYPE_KEYS[res.type])), ICONS[CATEGORY_ICONS[res.category]], esc(i18n.domain_name(lang, res.category)),
                        ('<span class="rf-context">%s</span>' % esc(t(lang, "rf_contextual"))) if entry["contextual"] else ""))
                    ui.html('<h3 class="rf-title" id="res-%s" dir="auto">%s</h3><p class="rf-desc" dir="auto">%s</p>' % (res.id, esc(title_of(res, lang)), esc(text_of(res, "description", lang))))
                    meta: List[str] = []
                    if res.source_id:
                        pages_text = t(lang, "rf_pages", first=res.pages[0], last=res.pages[-1]) if len(res.pages) > 1 and res.type == "document" else ", ".join(t(lang, "rf_page", page=p) for p in res.pages)
                        meta.append('<span class="chip page">%s</span><span class="chip">%s</span><span class="chip verified">%s%s</span>' % (
                            esc(source_label(res.source_id)), esc(pages_text), ICONS["check"], esc(t(lang, "rf_in_corpus"))))
                    if meta:
                        ui.html('<div class="rf-meta">%s</div>' % "".join(meta))
                    if res.contacts:
                        rows = "".join('<li><span class="rf-contact-label">%s</span> <span class="rf-contact-value" dir="ltr">%s</span></li>' % (esc(c.label.rstrip(":")), esc(c.value)) for c in res.contacts)
                        ui.html('<div class="rf-contacts"><div class="rf-contacts-head">%s%s</div><ul>%s</ul></div>' % (ICONS["shield"], esc(t(lang, "rf_contact_as_printed")), rows))
                    elif res.type == "department":
                        ui.html('<p class="rf-contact-note">%s%s</p>' % (ICONS["info"], esc(text_of(res, "contact_note", lang))))
                    for link in res.links:
                        if safe_url(link.url) is None:
                            continue
                        note = t(lang, "rf_document_link") if link.kind == "document_link" else t(lang, "rf_outside_corpus", date=link.verified_on)
                        ui.html('<div class="rf-link">%s%s <span class="rf-link-note">· %s</span><br/><span class="rf-link-desc">%s</span></div>' % (
                            ICONS["globe"], anchor(link.url, link.label_ar if lang == "ar" else link.label_en), esc(note), esc(link.note_ar if lang == "ar" else link.note_en)))
                    related = [C.TASKS[tid] for tid in res.related_tasks if tid in C.TASKS]
                    if related:
                        ui.html('<div class="rf-related"><span class="lbl">%s</span>%s</div>' % (esc(t(lang, "rf_related")), "".join(
                            '<a class="chip task" href="/onboarding?task=%s&lang=%s#task-%s">%s</a>' % (tk.id, lang, tk.id, esc(title_of(tk, lang))) for tk in related)))
                    ui.button(t(lang, "ob_ask"), on_click=lambda _e, rid=res.id: ui.navigate.to(ask_url(state["task"], t(lang, "rf_resource_question", resource=title_of(C.RESOURCES[rid], lang))))) \
                        .props('flat no-caps dense icon=forum').classes("ob-btn small").mark("ask-resource")

    def on_query(value: Any) -> None:
        state["query"] = str(value or "")[:200]
        render_results()

    def pick_category(cat: str) -> None:
        state["category"] = cat
        for key, btn in page.refs.get("filters", {}).items():
            chat._safe(btn.props, 'aria-pressed="%s"' % ("true" if key == cat else "false"))
            chat._safe(btn.classes, add="active") if key == cat else chat._safe(btn.classes, remove="active")
        render_results()

    def clear_context() -> None:
        state["task"] = None
        box = page.refs.get("context")
        if box is not None:
            chat._safe(box.delete)
        render_results()

    with main:
        ui.html('<div class="ob-page-head"><h1 class="ob-h1">%s</h1><p class="ob-sub">%s</p></div>' % (esc(t(lang, "rf_title")), esc(t(lang, "rf_subtitle"))))
        if task is not None:
            with ui.element("div").classes("rf-context-box").props('role="status"') as box:
                page.refs["context"] = box
                ui.html('<span>%s<strong dir="auto">%s</strong></span>' % (esc(t(lang, "rf_context", task="")), esc(title_of(task, lang))))
                ui.button(t(lang, "rf_clear_context"), on_click=clear_context).props('flat no-caps dense').classes("ob-btn small").mark("clear-context")
                ui.html('<a class="ob-link" href="/onboarding?task=%s&lang=%s#task-%s">%s</a>' % (task.id, lang, task.id, esc(t(lang, "chat_open_task"))))
        with ui.element("div").classes("rf-search"):
            field = ui.input(label=t(lang, "rf_search"), placeholder=t(lang, "rf_placeholder"), value=initial_query, on_change=lambda e: on_query(e.value)) \
                .props('outlined dense clearable stack-label dir=auto debounce=250 aria-label="%s"' % esc(t(lang, "rf_search"), True)).classes("rf-input").mark("finder-search")
            page.refs["input"] = field
            page.refs["count"] = ui.html("").classes("rf-count").props('role="status" aria-live="polite"')
        with ui.element("div").classes("ob-filters").props('role="group" aria-label="%s"' % esc(t(lang, "ob_filter_aria"), True)):
            page.refs["filters"] = {}
            for cat in ("all",) + C.CATEGORIES:
                label = t(lang, "ob_filter_all") if cat == "all" else i18n.domain_name(lang, cat)
                btn = ui.button(label, on_click=lambda _e, c=cat: pick_category(c)).props('flat no-caps dense aria-pressed="%s"' % ("true" if cat == state["category"] else "false")) \
                    .classes("ob-filter" + (" active" if cat == state["category"] else "")).mark("rfilter-" + cat)
                page.refs["filters"][cat] = btn
        page.refs["results"] = ui.element("section").classes("rf-results").props('aria-label="%s"' % esc(t(lang, "rf_results_aria"), True))
        ui.html('<p class="ob-foot">%s</p>' % esc(t(lang, "rf_disclaimer")))
    render_results()


def register_pages() -> None:
    @ui.page("/onboarding")
    def onboarding_page(request: Request) -> None:
        build_dashboard(request)

    @ui.page("/onboarding/procedure/{procedure_id}")
    def procedure_page(request: Request, procedure_id: str) -> None:
        build_procedure(request, procedure_id)

    @ui.page("/resources")
    def resources_page(request: Request) -> None:
        build_finder(request)


__all__ = ["build_dashboard", "build_finder", "build_procedure", "register_pages"]
