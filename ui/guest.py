"""
The guest's own pages (post-freeze PF-15, D-126): conversation history (``/history``), guest profile (``/profile``) and
settings (``/settings``), on the shared shell.

Everything here reads and writes through the owner-scoped PF-13 store: a conversation id comes from this owner's own list
and is re-checked by the store before it is opened, renamed or deleted; opening a conversation records it as the active
one in the owner's preferences and returns to the chat, so the id never appears in a URL. Saved answers are never
re-asked: the chat replays them from their stored records. Every preference control changes the page and is stored at
once; a value the store does not accept is refused with a message, never stored half-way. The profile is a guest
profile: an optional display name (validated: no identifiers, no markup), the session dates the store already keeps,
progress and recent activity of this owner only. Nothing on these pages calls the model.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

from fastapi import Request
from nicegui import run, ui

from onboarding import catalog as C
from onboarding.progress import dashboard
from persistence import StoreError
from persistence.store import clean_display_name

from . import i18n
from .i18n import t
from .shell import ICONS, Page, esc, guest_chip_html, html_props, initial_of, same_page_with, time_html

STATUS_KEYS = {"not_started": "status_not_started", "in_progress": "status_in_progress", "completed": "status_completed", "not_applicable": "status_not_applicable"}


def _parse(iso: Any) -> Optional[_dt.datetime]:
    text = str(iso or "").rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return _dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def age_group(iso: Any, now: Optional[_dt.datetime] = None) -> str:
    """The history group of a timestamp by elapsed time (timezone-independent): day, week, month or older."""
    stamp = _parse(iso)
    now = now or _dt.datetime.utcnow()
    if stamp is None:
        return "older"
    age = now - stamp
    if age <= _dt.timedelta(hours=24):
        return "day"
    if age <= _dt.timedelta(days=7):
        return "week"
    if age <= _dt.timedelta(days=30):
        return "month"
    return "older"


GROUP_KEYS = (("day", "hs_group_day"), ("week", "hs_group_week"), ("month", "hs_group_month"), ("older", "hs_group_older"))


# =====================================================================================================================
# Conversation history
# =====================================================================================================================
def build_history(request: Request) -> None:
    page = Page(request, "history")
    lang = page.lang
    content = page.shell(t(lang, "hs_title"))
    store, owner = page.store(), page.owner_id()
    state: Dict[str, Any] = {"rows": [], "query": "", "active": (page.pref("extra") or {}).get("active_conversation")}

    def load() -> None:
        state["rows"] = []
        if store is not None and owner is not None:
            try:
                state["rows"] = store.list_conversations(owner)
            except StoreError as exc:
                page._chat._log.error("loading conversations failed: %s", type(exc).__name__)
                page.notify(t(lang, "history_load_failed"), "warn")

    def visible_rows() -> List[Dict[str, Any]]:
        query = " ".join(state["query"].split()).casefold()
        return [r for r in state["rows"] if not query or query in (r.get("title") or "").casefold()]

    async def open_conversation(cid: str) -> None:
        if store is None or owner is None:
            return
        try:
            if await run.io_bound(store.get_conversation, owner, cid) is None:
                return                                       # not this owner's: nothing happens
            await run.io_bound(lambda: store.set_extra(owner, active_conversation=cid))
        except StoreError as exc:
            page._chat._log.error("opening a conversation failed: %s", type(exc).__name__)
            page.notify(t(lang, "history_load_failed"), "warn")
            return
        ui.navigate.to("/")

    def ask_rename(row: Dict[str, Any]) -> None:
        dialog, card = page.dialog("rename-dialog")
        with card:
            ui.html('<h2 class="dialog-title">%s</h2>' % esc(t(lang, "rename_title")))
            field = ui.input(label=t(lang, "rename_label"), value=row.get("title") or "") \
                .props('outlined dense autofocus dir=auto maxlength=120 aria-label="%s"' % esc(t(lang, "rename_label"), True)).classes("w-full").mark("rename-input")
            with ui.element("div").classes("dialog-actions"):
                ui.button(t(lang, "cancel"), on_click=lambda: page.close_dialog(dialog)).props("flat no-caps").classes("btn-secondary").mark("dialog-cancel")

                async def _save() -> None:
                    title = " ".join(str(field.value or "").split())
                    if not title:
                        page.notify(t(lang, "rename_empty"), "warn")
                        return
                    try:
                        await run.io_bound(store.rename_conversation, owner, row["id"], title)
                    except StoreError as exc:
                        page._chat._log.error("renaming a conversation failed: %s", type(exc).__name__)
                        page.notify(t(lang, "save_failed"), "warn")
                        page.close_dialog(dialog)
                        return
                    page.close_dialog(dialog)
                    load()
                    render()
                    page.notify(t(lang, "renamed"), "ok")
                ui.button(t(lang, "rename_ok"), on_click=_save).props("unelevated no-caps").mark("rename-ok")
                field.on("keydown.enter", lambda _e: _save())
        dialog.open()

    def ask_delete(row: Dict[str, Any]) -> None:
        title = row.get("title") or t(lang, "new_conversation")

        async def _delete() -> None:
            try:
                await run.io_bound(store.delete_conversation, owner, row["id"])
                if state["active"] == row["id"]:
                    await run.io_bound(lambda: store.set_extra(owner, active_conversation=None))
                    state["active"] = None
            except StoreError as exc:
                page._chat._log.error("deleting a conversation failed: %s", type(exc).__name__)
                page.notify(t(lang, "save_failed"), "warn")
                return
            load()
            render()
            page.notify(t(lang, "deleted"), "ok")
        page.confirm(t(lang, "delete_title"), t(lang, "delete_text", title=title), t(lang, "delete_ok"), _delete, ok_marker="delete-ok")

    def render() -> None:
        container = page.refs.get("list")
        if container is None:
            return
        container.clear()
        rows = visible_rows()
        count = page.refs.get("count")
        if count is not None:
            page._chat._safe(setattr, count, "content", esc(t(lang, "hs_count", n=len(state["rows"]))))
        with container:
            if not state["rows"]:
                ui.html('<div class="empty-state"><div class="empty-icon">%s</div><h2>%s</h2><p>%s</p><a class="link-btn" href="/?new=1">%s<span>%s</span></a></div>' % (
                    ICONS["chat"], esc(t(lang, "hs_empty")), esc(t(lang, "hs_empty_hint")), ICONS["plus"], esc(t(lang, "hs_start"))))
                return
            if not rows:
                ui.html('<div class="ob-empty" role="status">%s<span>%s</span></div>' % (ICONS["info"], esc(t(lang, "hs_no_match", q=state["query"]))))
                return
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for row in rows:
                groups.setdefault(age_group(row.get("updated_at")), []).append(row)
            for key, label_key in GROUP_KEYS:
                if key not in groups:
                    continue
                with ui.element("section").classes("hs-group").props('aria-label="%s"' % esc(t(lang, label_key), True)):
                    ui.html('<h2 class="hs-group-title">%s</h2>' % esc(t(lang, label_key)))
                    with ui.element("div").classes("hs-list"):
                        for row in groups[key]:
                            title = row.get("title") or t(lang, "new_conversation")
                            current = row["id"] == state["active"]
                            with ui.element("article").classes("hs-row" + (" current" if current else "")).mark("hs-row"):
                                ui.html('<div class="hs-icon" aria-hidden="true">%s</div>' % ICONS["chat"])
                                with ui.element("div").classes("hs-main"):
                                    btn = ui.button(on_click=lambda _e, cid=row["id"]: open_conversation(cid)) \
                                        .props('flat no-caps align=left aria-label="%s"' % esc(t(lang, "hs_open_aria", title=title), True)).classes("hs-title").mark("hs-open")
                                    with btn:
                                        ui.html('<span class="hs-title-text" dir="auto">%s</span>' % esc(title))
                                    meta = [time_html(row.get("updated_at"), "hs-time"), esc(t(lang, "hs_messages", n=row.get("message_count") or 0))]
                                    task = C.task(row.get("context_task")) if row.get("context_task") else None
                                    if task is not None:
                                        meta.append('<span class="chip task" title="%s">%s</span>' % (esc(t(lang, "hs_task"), True), esc(task.title_ar if lang == "ar" else task.title_en)))
                                    if current:
                                        meta.append('<span class="badge">%s</span>' % esc(t(lang, "hs_current")))
                                    ui.html('<div class="hs-meta">%s</div>' % "".join('<span>%s</span>' % m for m in meta if m))
                                with ui.element("div").classes("hs-actions"):
                                    rename = ui.button(on_click=lambda _e, r=row: ask_rename(r)).props('flat dense aria-label="%s" title="%s"' % (
                                        esc(t(lang, "hs_rename_aria", title=title), True), esc(t(lang, "rename"), True))).classes("icon-action").mark("hs-rename")
                                    with rename:
                                        ui.html(ICONS["pencil"])
                                    delete = ui.button(on_click=lambda _e, r=row: ask_delete(r)).props('flat dense aria-label="%s" title="%s"' % (
                                        esc(t(lang, "hs_delete_aria", title=title), True), esc(t(lang, "delete"), True))).classes("icon-action danger").mark("hs-delete")
                                    with delete:
                                        ui.html(ICONS["trash"])

    def on_search(value: Any) -> None:
        state["query"] = str(value or "")[:120]
        render()

    load()
    with content:
        ui.html('<div class="page-head"><div class="page-head-text"><h1 class="page-title">%s</h1><p class="page-sub">%s</p></div></div>' % (
            esc(t(lang, "hs_title")), esc(t(lang, "hs_subtitle"))))
        with ui.element("div").classes("toolbar"):
            ui.input(label=t(lang, "hs_search"), placeholder=t(lang, "hs_search_placeholder"), on_change=lambda e: on_search(e.value)) \
                .props('outlined dense clearable stack-label dir=auto debounce=200 aria-label="%s"' % esc(t(lang, "hs_search"), True)).mark("history-search")
            page.refs["count"] = ui.html("").classes("rf-count").props('role="status" aria-live="polite"')
            ui.html('<a class="link-btn" href="/?new=1">%s<span>%s</span></a>' % (ICONS["plus"], esc(t(lang, "new_chat"))))
        page.refs["list"] = ui.element("div").classes("hs-groups")
    render()


# =====================================================================================================================
# Guest profile
# =====================================================================================================================
def _lang_name(lang: str, value: Optional[str], auto_key: str = "pf_value_auto") -> str:
    return {"en": "English", "ar": "العربية"}.get(value or "", t(lang, auto_key))


def build_profile(request: Request) -> None:
    page = Page(request, "profile")
    lang = page.lang
    content = page.shell(t(lang, "pf_title"))
    store, owner = page.store(), page.owner_id()
    info: Dict[str, Any] = {}
    progress: Dict[str, Dict[str, Any]] = {}
    events: List[Dict[str, Any]] = []
    conversations: List[Dict[str, Any]] = []
    if store is not None and owner is not None:
        try:
            info = store.owner_info(owner)
            progress = store.task_progress(owner)
            events = store.progress_events(owner, 6)
            conversations = store.list_conversations(owner, limit=3)
        except StoreError as exc:
            page._chat._log.error("loading the profile failed: %s", type(exc).__name__)
    d = dashboard(progress, C.tasks_by_category())

    def hero_html() -> str:
        name = page.pref("display_name")
        facts = []
        if info.get("created_at"):
            facts.append('<dt>%s</dt><dd>%s</dd>' % (esc(t(lang, "pf_saving_since")), time_html(info["created_at"], "pf-time", full=True)))
            facts.append('<dt>%s</dt><dd>%s</dd>' % (esc(t(lang, "pf_last_seen")), time_html(info.get("last_seen_at"), "pf-time", full=True)))
        else:
            facts.append('<dt>%s</dt><dd>%s</dd>' % (esc(t(lang, "pf_status")), esc(t(lang, "pf_nothing_saved"))))
        return ('<div class="profile-hero"><span class="guest-avatar large" aria-hidden="true">%s</span><div><h2 class="profile-name" dir="auto">%s</h2>'
                '<span class="badge">%s%s</span></div></div><dl class="facts">%s</dl>') % (
            esc(initial_of(name)), esc(name or t(lang, "guest_label")), ICONS["shield"], esc(t(lang, "guest_badge")), "".join(facts))

    def refresh_identity() -> None:
        hero = page.refs.get("hero")
        if hero is not None:
            page._chat._safe(setattr, hero, "content", hero_html())
        chip = page.refs.get("chip")
        if chip is not None:
            page._chat._safe(setattr, chip, "content", guest_chip_html(lang, page.pref("display_name")))

    async def save_name() -> None:
        raw = page.refs["name"].value
        try:
            cleaned = clean_display_name(raw)
        except StoreError:
            page.notify(t(lang, "pf_name_invalid"), "warn")
            return
        if cleaned is None:
            await remove_name()
            return
        if await page.save_pref(display_name=cleaned):
            page.refs["name"].value = cleaned
            refresh_identity()
            page.notify(t(lang, "pf_name_saved"), "ok")
        else:
            page.notify(t(lang, "st_save_failed"), "warn")

    async def remove_name() -> None:
        if page.owner_id() is None:
            page.refs["name"].value = ""
            return
        if await page.save_pref(display_name=None):
            page.refs["name"].value = ""
            refresh_identity()
            page.notify(t(lang, "pf_name_removed"), "ok")
        else:
            page.notify(t(lang, "st_save_failed"), "warn")

    with content:
        ui.html('<div class="page-head"><div class="page-head-text"><h1 class="page-title">%s</h1><p class="page-sub">%s</p></div></div>' % (
            esc(t(lang, "pf_title")), esc(t(lang, "pf_subtitle"))))
        with ui.element("div").classes("grid-2"):
            with ui.element("section").classes("card").props('aria-label="%s"' % esc(t(lang, "pf_status"), True)):
                page.refs["hero"] = ui.html(hero_html())
                ui.html('<div class="setting-label" style="margin-top:16px">%s <span class="muted">(%s)</span></div><p class="setting-help">%s</p>' % (
                    esc(t(lang, "pf_name")), esc(t(lang, "pf_name_optional")), esc(t(lang, "pf_name_help"))))
                with ui.element("div").classes("name-form"):
                    page.refs["name"] = ui.input(label=t(lang, "pf_name"), placeholder=t(lang, "pf_name_placeholder"), value=page.pref("display_name") or "") \
                        .props('outlined dense stack-label dir=auto maxlength=40 counter aria-label="%s"' % esc(t(lang, "pf_name"), True)).mark("profile-name")
                    ui.button(t(lang, "pf_save_name"), on_click=save_name).props("unelevated no-caps").classes("btn-primary").mark("profile-save")
                    ui.button(t(lang, "pf_remove_name"), on_click=remove_name).props("outline no-caps").classes("btn-secondary").mark("profile-remove")
                page.refs["name"].on("keydown.enter", lambda _e: save_name())
            with ui.element("section").classes("card").props('aria-label="%s"' % esc(t(lang, "pf_progress"), True)):
                ui.html('<div class="card-head"><div class="card-icon">%s</div><div><h2 class="card-title">%s</h2><p class="card-desc">%s</p></div></div>'
                        '<div class="progress-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="%d" aria-label="%s"><span style="width:%d%%"></span></div>'
                        '<dl class="facts"><dt>%s</dt><dd data-stat="completed">%d</dd><dt>%s</dt><dd data-stat="in_progress">%d</dd><dt>%s</dt><dd data-stat="remaining">%d</dd></dl>'
                        '<p style="margin-top:12px"><a class="quiet-link" href="/onboarding">%s<span>%s</span></a></p>' % (
                            ICONS["list_checks"], esc(t(lang, "pf_progress")), esc(t(lang, "ob_percent", percent=d["percent"])), d["percent"],
                            esc(t(lang, "ob_percent", percent=d["percent"]), True), d["percent"],
                            esc(t(lang, "ob_completed")), d["completed"], esc(t(lang, "ob_in_progress")), d["in_progress"], esc(t(lang, "ob_remaining")), d["remaining"],
                            ICONS["arrow"], esc(t(lang, "st_open_checklist"))))
            prefs = page.ctx.get("prefs") or {}
            rows = [(t(lang, "st_interface_lang"), _lang_name(lang, lang)), (t(lang, "st_answer_lang"), _lang_name(lang, prefs.get("answer_lang"))),
                    (t(lang, "st_voice_lang"), _lang_name(lang, prefs.get("voice_lang"))),
                    (t(lang, "st_theme"), t(lang, "theme_dark" if page.ctx["theme"] == "dark" else "theme_light")),
                    (t(lang, "st_text_size"), t(lang, "st_text_large" if prefs.get("text_size") == "large" else "st_text_standard")),
                    (t(lang, "st_motion"), t(lang, "st_motion_on" if prefs.get("reduced_motion") == "on" else "st_motion_off"))]
            with ui.element("section").classes("card").props('aria-label="%s"' % esc(t(lang, "pf_preferences"), True)):
                ui.html('<div class="card-head"><div class="card-icon">%s</div><div><h2 class="card-title">%s</h2></div></div><dl class="facts">%s</dl>'
                        '<p style="margin-top:12px"><a class="quiet-link" href="/settings">%s<span>%s</span></a></p>' % (
                            ICONS["sliders"], esc(t(lang, "pf_preferences")), "".join('<dt>%s</dt><dd>%s</dd>' % (esc(k), esc(v)) for k, v in rows),
                            ICONS["arrow"], esc(t(lang, "pf_edit_settings"))))
            items: List[str] = []
            for e in events:
                task = C.task(e.get("task_id"))
                if task is None:
                    continue
                items.append('<li><span class="dot" aria-hidden="true"></span><div><div dir="auto">%s</div>%s</div></li>' % (
                    esc(t(lang, "pf_event", task=task.title_ar if lang == "ar" else task.title_en, status=t(lang, STATUS_KEYS.get(e.get("to_status"), "status_not_started")))),
                    time_html(e.get("created_at"), "pf-time")))
            for conv in conversations:
                items.append('<li><span class="dot conv" aria-hidden="true"></span><div><div dir="auto">%s: <a class="quiet-link" href="/history">%s</a></div>%s</div></li>' % (
                    esc(t(lang, "pf_conversation")), esc(conv.get("title") or t(lang, "new_conversation")), time_html(conv.get("updated_at"), "pf-time")))
            with ui.element("section").classes("card").props('aria-label="%s"' % esc(t(lang, "pf_recent"), True)):
                ui.html('<div class="card-head"><div class="card-icon">%s</div><div><h2 class="card-title">%s</h2></div></div>%s' % (
                    ICONS["history"], esc(t(lang, "pf_recent")), ('<ul class="activity">%s</ul>' % "".join(items)) if items else '<p class="card-desc" style="margin-top:8px">%s</p>' % esc(t(lang, "pf_no_activity"))))
        ui.html('<section class="card" style="margin-top:16px" aria-label="%s"><div class="card-head"><div class="card-icon">%s</div><div><h2 class="card-title">%s</h2>'
                '<p class="card-desc">%s</p></div></div></section>' % (esc(t(lang, "pf_about"), True), ICONS["info"], esc(t(lang, "pf_about")), esc(t(lang, "pf_about_text"))))


# =====================================================================================================================
# Settings
# =====================================================================================================================
def build_settings(request: Request) -> None:
    page = Page(request, "settings")
    lang = page.lang
    content = page.shell(t(lang, "st_title"))
    store, owner = page.store(), page.owner_id()
    prefs = page.ctx.get("prefs") or {}
    counts = {"conversations": 0}
    progress: Dict[str, Dict[str, Any]] = {}
    if store is not None and owner is not None:
        try:
            counts["conversations"] = len(store.list_conversations(owner))
            progress = store.task_progress(owner)
        except StoreError as exc:
            page._chat._log.error("loading settings data failed: %s", type(exc).__name__)
    d = dashboard(progress, C.tasks_by_category())

    async def save(**changes: Any) -> None:
        if await page.save_pref(**changes):
            page.notify(t(lang, "st_saved"), "ok")
        else:
            page.notify(t(lang, "st_save_failed"), "warn")

    async def on_interface(value: Any) -> None:
        if value not in i18n.LANGUAGES or value == lang:
            return
        if await page.save_pref(interface_lang=value):
            ui.navigate.to(same_page_with(request, lang=value))
        else:
            page.notify(t(lang, "st_save_failed"), "warn")

    async def on_theme(value: Any) -> None:
        if value not in ("light", "dark"):
            return
        page.ctx["theme"] = value
        page.apply_html()
        await save(theme=value)

    async def on_text(value: Any) -> None:
        if value not in ("standard", "large"):
            return
        page.ctx.setdefault("prefs", {})["text_size"] = value
        page.apply_html()
        await save(text_size=value)

    async def on_motion(value: Any) -> None:
        if value not in ("off", "on"):
            return
        page.ctx.setdefault("prefs", {})["reduced_motion"] = value
        page.apply_html()
        await save(reduced_motion=value)

    async def reset() -> None:
        owner_now = page.owner_id()
        if store is not None and owner_now is not None:
            try:
                await run.io_bound(store.reset_preferences, owner_now)
            except StoreError as exc:
                page._chat._log.error("resetting preferences failed: %s", type(exc).__name__)
                page.notify(t(lang, "st_save_failed"), "warn")
                return
        ui.navigate.to("/settings")

    async def delete_all() -> None:
        owner_now = page.owner_id()
        if store is None or owner_now is None:
            return
        try:
            await run.io_bound(store.delete_all_conversations, owner_now)
            await run.io_bound(lambda: store.set_extra(owner_now, active_conversation=None))
        except StoreError as exc:
            page._chat._log.error("deleting the conversations failed: %s", type(exc).__name__)
            page.notify(t(lang, "save_failed"), "warn")
            return
        counts["conversations"] = 0
        el = page.refs.get("conv_count")
        if el is not None:
            page._chat._safe(setattr, el, "content", esc(t(lang, "st_conversations_count", n=0)))
        page.notify(t(lang, "all_deleted"), "ok")

    def section(icon: str, title_key: str, desc_key: str):
        card = ui.element("section").classes("card").props('aria-labelledby="st-%s"' % title_key)
        with card:
            ui.html('<div class="card-head"><div class="card-icon">%s</div><div><h2 class="card-title" id="st-%s">%s</h2><p class="card-desc">%s</p></div></div>' % (
                ICONS[icon], title_key, esc(t(lang, title_key)), esc(t(lang, desc_key))))
        return card

    def row(label_key: str, help_key: Optional[str], options: Dict[str, str], value: str, handler, marker: str) -> None:
        with ui.element("div").classes("setting-row"):
            ui.html('<div><div class="setting-label">%s</div>%s</div>' % (esc(t(lang, label_key)), ('<div class="setting-help">%s</div>' % esc(t(lang, help_key))) if help_key else ""))
            ui.toggle(options, value=value, on_change=lambda e: handler(e.value)) \
                .props('no-caps unelevated toggle-color=primary aria-label="%s"' % esc(t(lang, label_key), True)).classes("seg").mark(marker)

    with content:
        ui.html('<div class="page-head"><div class="page-head-text"><h1 class="page-title">%s</h1><p class="page-sub">%s</p></div></div>' % (
            esc(t(lang, "st_title")), esc(t(lang, "st_subtitle"))))
        with section("globe", "st_language", "st_language_desc"):
            row("st_interface_lang", None, {"en": "English", "ar": "العربية"}, lang, on_interface, "pref-lang")
            row("st_answer_lang", "st_answer_lang_help", {"auto": t(lang, "pf_value_auto"), "en": "English", "ar": "العربية"},
                prefs.get("answer_lang") or "auto", lambda v: save(answer_lang=v if v in i18n.ANSWER_LANGUAGES else "auto"), "pref-answer-lang")
            row("st_voice_lang", "st_voice_lang_help", {"auto": t(lang, "pf_value_auto"), "en": "English", "ar": "العربية"},
                prefs.get("voice_lang") or "auto", lambda v: save(voice_lang=v if v in i18n.VOICE_LANGUAGES else "auto"), "pref-voice-lang")
        with section("sun", "st_appearance", "st_appearance_desc"):
            row("st_theme", None, {"light": t(lang, "theme_light"), "dark": t(lang, "theme_dark")}, page.ctx["theme"], on_theme, "pref-theme")
        with section("user", "st_accessibility", "st_accessibility_desc"):
            row("st_text_size", None, {"standard": t(lang, "st_text_standard"), "large": t(lang, "st_text_large")}, prefs.get("text_size") or "standard", on_text, "pref-text-size")
            row("st_motion", "st_motion_help", {"off": t(lang, "st_motion_off"), "on": t(lang, "st_motion_on")}, prefs.get("reduced_motion") or "off", on_motion, "pref-motion")
        with section("shield", "st_data", "st_data_desc"):
            with ui.element("div").classes("data-row"):
                page.refs["conv_count"] = ui.html(esc(t(lang, "st_conversations_count", n=counts["conversations"])))
                ui.html('<a class="quiet-link" href="/history">%s<span>%s</span></a>' % (ICONS["history"], esc(t(lang, "st_open_history"))))
            with ui.element("div").classes("data-row"):
                ui.html(esc(t(lang, "st_progress_count", done=d["completed"], total=d["total"])))
                ui.html('<a class="quiet-link" href="/onboarding">%s<span>%s</span></a>' % (ICONS["list_checks"], esc(t(lang, "st_open_checklist"))))
            with ui.element("div").classes("data-actions"):
                ui.button(t(lang, "st_reset"), on_click=lambda: page.confirm(t(lang, "st_reset_title"), t(lang, "st_reset_text"), t(lang, "st_reset_ok"), reset,
                                                                         ok_marker="reset-ok", danger=False)) \
                    .props("outline no-caps").classes("btn-secondary").mark("settings-page-reset")
                ui.button(t(lang, "clear_all"), on_click=lambda: page.confirm(t(lang, "clear_all_title"), t(lang, "clear_all_text"), t(lang, "clear_all_ok"), delete_all,
                                                                         ok_marker="clear-all-ok")) \
                    .props("outline no-caps").classes("btn-danger").mark("settings-page-clear")


def register_pages() -> None:
    @ui.page("/history")
    def history_page(request: Request) -> None:
        build_history(request)

    @ui.page("/profile")
    def profile_page(request: Request) -> None:
        build_profile(request)

    @ui.page("/settings")
    def settings_page(request: Request) -> None:
        build_settings(request)


__all__ = ["age_group", "build_history", "build_profile", "build_settings", "register_pages"]
