"""
Design system of the chat page (post-freeze PF-10).

One accent (the University of Sharjah green of the approved logo), a light theme by default, a dark theme on request,
system typography with an Arabic-capable fallback, logical (direction-aware) spacing so the layout can host right-to-left
text later, visible keyboard focus, reduced-motion support, and responsive rules at 900 px (the sidebar gives way to a
single column) and 540 px (phone: the topbar carries New chat, D-10).

Tokens live on ``:root`` and are redefined under ``[data-theme="dark"]``; components only ever use tokens.
"""

GLOBAL_CSS = """
<style>
/* ================================================================
   TOKENS
   ================================================================ */
:root {
    --brand:        #0B7A52;   /* UoS green, darkened for AA contrast on white text */
    --brand-strong: #075C3D;
    --brand-text:   #0A6B48;
    --brand-soft:   #E7F3ED;
    --brand-ring:   rgba(11, 122, 82, 0.35);
    --brand-fill:   #0B7A52;   /* filled controls: white text reads at 5.4:1 */
    --brand-fill-strong: #075C3D;

    --bg:           #F3F5F4;
    --surface:      #FFFFFF;
    --surface-2:    #F7F9F8;
    --surface-3:    #EDF1EF;
    --text:         #182320;
    --text-2:       #46524D;
    --text-3:       #65716C;
    --border:       #DCE3DF;
    --border-strong:#C3CDC7;

    --user-bubble:  #0B7A52;
    --user-text:    #FFFFFF;

    --ok:           #0B7A52;
    --warn:         #8A5A00;
    --warn-soft:    #FFF4D6;
    --danger:       #B3261E;
    --danger-soft:  #FDECEA;
    --info-soft:    #EEF3FF;

    --shadow-sm: 0 1px 2px rgba(16, 32, 24, 0.06);
    --shadow-md: 0 6px 20px rgba(16, 32, 24, 0.08);

    --sans: "Inter", "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Noto Sans Arabic", "Noto Naskh Arabic", Tahoma, sans-serif;
    --sans-ar: "Segoe UI", "Noto Naskh Arabic", "Noto Sans Arabic", "Noto Kufi Arabic", Tahoma, "Arabic Typesetting", Arial, sans-serif;
    --mono: ui-monospace, "Cascadia Code", "SF Mono", Menlo, Consolas, monospace;

    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-pill: 999px;
    --sidebar-w: 288px;
    --content-w: 820px;
    --transition: 160ms ease;
}
[data-theme="dark"] {
    --brand:        #2EA97C;
    --brand-strong: #249268;
    --brand-text:   #49C291;
    --brand-soft:   rgba(46, 169, 124, 0.16);
    --brand-ring:   rgba(73, 194, 145, 0.45);
    --brand-fill:   #0F7A54;   /* darker fill than the dark accent so white text keeps 5.3:1 */
    --brand-fill-strong: #0C6243;

    --bg:           #0F1513;
    --surface:      #161D1A;
    --surface-2:    #1C2420;
    --surface-3:    #232D28;
    --text:         #E9EFEC;
    --text-2:       #B9C4BE;
    --text-3:       #8F9C96;
    --border:       #283330;
    --border-strong:#36433E;

    --user-bubble:  #0F7A54;
    --user-text:    #FFFFFF;

    --ok:           #49C291;
    --warn:         #F0C36B;
    --warn-soft:    rgba(240, 195, 107, 0.14);
    --danger:       #F28B82;
    --danger-soft:  rgba(242, 139, 130, 0.14);
    --info-soft:    rgba(120, 160, 255, 0.12);

    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.35);
    --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.45);
}

/* ================================================================
   BASE
   ================================================================ */
html, body {
    height: 100%;
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 15px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}
body { overflow: hidden; }
.nicegui-content { padding: 0 !important; gap: 0 !important; max-width: none !important; height: 100%; }
* { box-sizing: border-box; }
h1, h2, h3, p, ul, ol { margin: 0; }
h1, h2, h3 { line-height: 1.3; letter-spacing: 0; font-weight: 650; }   /* Quasar's display sizes do not apply here */
button { font-family: inherit; }
::selection { background: var(--brand-soft); }
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 10px; border: 2px solid transparent; background-clip: padding-box; }
*::-webkit-scrollbar-track { background: transparent; }

/* The page root fills the viewport whatever the shell's intrinsic width (the shell is rebuilt inside it on a language switch) */
.app-root { width: 100%; height: 100%; min-width: 0; display: block; }

/* Real right-to-left layout: Quasar's stylesheet forces ``direction: ltr`` on html and body, which would silently turn
   the dir="rtl" attribute into text-level bidi only; the attribute selectors below outrank that rule (post-freeze PF-11) */
html[dir="rtl"], html[dir="rtl"] body, html[dir="rtl"] #q-app, html[dir="rtl"] .q-layout, html[dir="rtl"] .q-page-container, html[dir="rtl"] .q-page { direction: rtl; }

/* Arabic interface and Arabic text: system Arabic fonts, a little more leading, the same tokens (post-freeze PF-11) */
html[lang="ar"] body, html[lang="ar"] button, html[lang="ar"] textarea, html[lang="ar"] input { font-family: var(--sans-ar); }
html[lang="ar"] .bubble, html[lang="ar"] .hero p, html[lang="ar"] .about, html[lang="ar"] .topic-blurb { line-height: 1.8; }
.bubble[dir="auto"]:lang(ar), .answer [dir="auto"] { unicode-bidi: plaintext; }
.answer p[dir="auto"], .answer li[dir="auto"], .answer blockquote[dir="auto"] { text-align: start; }
.answer blockquote[dir="auto"] { unicode-bidi: isolate; }
.topbar-lang.q-btn { min-width: 44px; padding: 0 12px; font-weight: 650; font-size: 13.5px; letter-spacing: 0; }
.topbar-lang.q-btn .q-btn__content { gap: 6px; }
.answer-lang { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-3); }
.answer-lang .q-field { min-width: 0; }
.answer-lang .q-field__native, .answer-lang .q-field__control { min-height: 26px !important; font-size: 12.5px; color: var(--text-2); }
.answer-lang .q-field__control::before { border-color: var(--border) !important; }
.answer-lang .q-select__dropdown-icon { color: var(--text-3); font-size: 18px; }
.translation-note { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-3); }
.translation-note svg { width: 14px; height: 14px; flex: none; }
details.original { border: 1px dashed var(--border); border-radius: var(--radius-md); background: var(--surface-2); }
details.original summary { cursor: pointer; padding: 8px 14px; font-size: 13px; font-weight: 600; color: var(--text-2); list-style: none; display: flex; align-items: center; gap: 8px; }
details.original summary::-webkit-details-marker { display: none; }
details.original summary svg { width: 16px; height: 16px; transition: transform var(--transition); }
details.original[open] summary svg { transform: rotate(90deg); }
html[dir="rtl"] details.original summary svg, html[dir="rtl"] details.evidence summary svg { transform: rotate(180deg); }
html[dir="rtl"] details.original[open] summary svg, html[dir="rtl"] details.evidence[open] summary svg { transform: rotate(90deg); }
.original-body { padding: 4px 14px 12px 14px; font-size: 14px; color: var(--text); }
html[dir="rtl"] .example-btn .q-icon, html[dir="rtl"] .topic-title svg.arrow { transform: scaleX(-1); }
html[dir="rtl"] .q-btn .q-icon.on-left { margin-right: 0; margin-left: 8px; }
html[dir="rtl"] .composer-hint kbd { direction: ltr; unicode-bidi: isolate; }

/* keyboard focus: one visible ring everywhere */
:focus-visible { outline: 2px solid var(--brand); outline-offset: 2px; border-radius: var(--radius-sm); }
.q-btn:focus-visible, button:focus-visible, summary:focus-visible, a:focus-visible, textarea:focus-visible { outline: 2px solid var(--brand); outline-offset: 2px; }
/* Quasar marks every button ``no-outline`` (outline: 0 !important) and we hide its own focus helper, so the ring must win here */
.q-btn.no-outline:focus-visible, .q-btn:focus-visible { outline: 2px solid var(--brand) !important; outline-offset: 2px !important; }
.q-focus-helper { display: none !important; }
.q-ripple { display: none !important; }

.sr-only { position: absolute !important; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0; }
.skip-link { position: absolute; inset-inline-start: 12px; top: -48px; z-index: 50; padding: 8px 14px; background: var(--brand-fill); color: #fff; border-radius: var(--radius-sm); font-weight: 600; text-decoration: none; transition: top var(--transition); }
.skip-link:focus { top: 12px; }

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; scroll-behavior: auto !important; }
}

/* Quasar buttons inherit our colours; no default blue */
.q-btn { color: inherit; text-transform: none; letter-spacing: 0; }
.q-btn .q-btn__content { gap: 8px; }
.q-btn svg { width: 18px; height: 18px; stroke-width: 2; }
.btn-inner { display: inline-flex; align-items: center; gap: 8px; min-width: 0; }
.btn-inner > span { min-width: 0; }

/* ================================================================
   SHELL
   ================================================================ */
.app-shell { display: flex; height: 100vh; height: 100dvh; width: 100%; background: var(--bg); overflow: hidden; }

.sidebar {
    width: var(--sidebar-w); min-width: var(--sidebar-w);
    display: flex; flex-direction: column; gap: 14px;
    padding: 18px 16px 16px 16px;
    background: var(--surface);
    border-inline-end: 1px solid var(--border);
    overflow-y: auto;
}
.brand { display: flex; align-items: center; gap: 12px; padding: 2px 4px 10px 4px; border-bottom: 1px solid var(--border); }
.brand-logo { width: 44px; height: 44px; flex: none; display: grid; place-items: center; border-radius: var(--radius-md); background: #fff; border: 1px solid var(--border); overflow: hidden; }
.brand-logo img { width: 40px; height: 40px; object-fit: contain; }
.brand-text { min-width: 0; }
.brand-org { font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--brand-text); }
.brand-name { font-size: 14.5px; font-weight: 650; color: var(--text); line-height: 1.3; }

.new-chat-btn.q-btn {
    width: 100%; min-height: 44px; border-radius: var(--radius-md);
    background: var(--brand-fill) !important; color: #fff !important; font-weight: 600; font-size: 14px;
    box-shadow: var(--shadow-sm);
}
.new-chat-btn.q-btn:hover { background: var(--brand-fill-strong) !important; }
.new-chat-btn .q-btn__content { justify-content: center; }
.new-chat-btn .q-icon { font-size: 20px; }

.sb-label { font-size: 11.5px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-3); padding: 4px 6px 0 6px; }
.sb-note { font-size: 12.5px; color: var(--text-3); padding: 0 6px; line-height: 1.5; }
.session-list { display: flex; flex-direction: column; gap: 4px; }
.session-item.q-btn {
    width: 100%; min-height: 40px; justify-content: flex-start; text-align: start;
    padding: 8px 12px; border-radius: var(--radius-sm); font-size: 13.5px; font-weight: 500; color: var(--text-2);
    border: 1px solid transparent;
}
.session-item.q-btn .q-btn__content { justify-content: flex-start; width: 100%; }
.session-item .session-title { display: block; width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-align: start; }
.session-item.q-btn:hover { background: var(--surface-2); color: var(--text); }
.session-item.active.q-btn { background: var(--brand-soft); color: var(--brand-text); border-color: transparent; font-weight: 600; }

.sb-footer { margin-top: auto; display: flex; flex-direction: column; gap: 10px; padding-top: 12px; border-top: 1px solid var(--border); }
.about { font-size: 12.5px; color: var(--text-2); line-height: 1.55; padding: 0 4px; }
.about strong { color: var(--text); font-weight: 600; }

/* ================================================================
   MAIN COLUMN
   ================================================================ */
.main { flex: 1; min-width: 0; display: flex; flex-direction: column; position: relative; background: var(--bg); }

.topbar {
    height: 60px; flex: none; display: flex; align-items: center; justify-content: space-between; gap: 12px;
    padding: 0 20px; background: var(--surface); border-bottom: 1px solid var(--border);
}
.topbar-left { display: flex; align-items: center; gap: 12px; min-width: 0; }
.tl { flex: 1 1 auto; min-width: 0; }
.topbar-left > div { min-width: 0; display: flex; flex-direction: column; justify-content: center; }
.topbar-brand { display: none; width: 34px; height: 34px; border-radius: var(--radius-sm); background: #fff; border: 1px solid var(--border); place-items: center; flex: none; }
.topbar-brand img { width: 30px; height: 30px; object-fit: contain; }
.topbar-title { font-size: 16px; font-weight: 650; line-height: 1.25; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.topbar-sub { font-size: 12.5px; line-height: 1.25; color: var(--text-3); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.topbar-right { display: flex; align-items: center; gap: 8px; flex: none; }

.status-pill {
    display: inline-flex; align-items: center; gap: 8px; height: 34px; padding: 0 12px;
    border-radius: var(--radius-pill); border: 1px solid var(--border); background: var(--surface-2);
    font-size: 12.5px; font-weight: 600; color: var(--text-2); white-space: nowrap;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-3); flex: none; }
.status-pill.ok .status-dot { background: var(--ok); box-shadow: 0 0 0 3px var(--brand-soft); }
.status-pill.warn { background: var(--warn-soft); border-color: transparent; color: var(--warn); }
.status-pill.warn .status-dot { background: var(--warn); }
.status-pill.danger { background: var(--danger-soft); border-color: transparent; color: var(--danger); }
.status-pill.danger .status-dot { background: var(--danger); }

.topbar-btn.q-btn {
    width: 40px; height: 40px; min-height: 40px; padding: 0; border-radius: var(--radius-sm);
    border: 1px solid var(--border); background: var(--surface); color: var(--text-2);
}
.topbar-btn.q-btn:hover { background: var(--surface-2); color: var(--brand-text); border-color: var(--border-strong); }
.topbar-btn .q-btn__content, .topbar-btn svg { color: inherit !important; }
.topbar-btn svg { width: 18px; height: 18px; }
.topbar-new-chat { display: none; }
.topbar-btn .sun  { display: none; }
.topbar-btn .moon { display: block; }
[data-theme="dark"] .topbar-btn .sun  { display: block; }
[data-theme="dark"] .topbar-btn .moon { display: none; }

.dev-toggle { display: inline-flex; align-items: center; gap: 6px; height: 34px; padding: 0 6px 0 12px; border-radius: var(--radius-pill); border: 1px dashed var(--border-strong); font-size: 12.5px; color: var(--text-3); }
.dev-toggle .q-toggle { padding: 0 !important; min-height: 0 !important; }
.q-toggle__inner--truthy { color: var(--brand) !important; }
.q-toggle__inner--falsy { color: var(--text-3) !important; }

/* ================================================================
   CONVERSATION
   ================================================================ */
.chat-scroll { flex: 1; min-height: 0; overflow-y: auto; overflow-x: hidden; scroll-behavior: smooth; }
.chat-inner { max-width: var(--content-w); margin: 0 auto; padding: 28px 24px 24px 24px; display: flex; flex-direction: column; gap: 22px; }

.msg-row { display: flex; gap: 12px; align-items: flex-start; animation: fadeUp 220ms ease both; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.msg-row.user { justify-content: flex-end; }
.msg-row.user > .nicegui-html { max-width: 72%; }
.msg-row.assistant > .assistant-wrap { flex: 1; min-width: 0; max-width: 100%; }

.avatar { width: 34px; height: 34px; flex: none; border-radius: 50%; display: grid; place-items: center; border: 1px solid var(--border); background: #fff; overflow: hidden; margin-top: 2px; }
.avatar img { width: 26px; height: 26px; object-fit: contain; }
.avatar.user { background: var(--surface-3); color: var(--text-2); font-size: 11px; font-weight: 700; letter-spacing: 0.04em; }

.bubble { border-radius: var(--radius-lg); font-size: 15px; line-height: 1.65; overflow-wrap: anywhere; }
.bubble.user {
    background: var(--user-bubble); color: var(--user-text);
    padding: 12px 16px; border-end-end-radius: 6px; white-space: pre-wrap; box-shadow: var(--shadow-sm);
}
.assistant-wrap { display: flex; flex-direction: column; gap: 10px; }
.assistant-name { font-size: 12px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--text-3); }
.bubble.assistant {
    background: var(--surface); color: var(--text); border: 1px solid var(--border);
    padding: 16px 18px; border-start-start-radius: 6px; box-shadow: var(--shadow-sm);
}

/* answer typography (rendered from sanitised Markdown) */
.answer p { margin: 0 0 10px 0; }
.answer p:last-child { margin-bottom: 0; }
.answer ul, .answer ol { margin: 6px 0 10px 0; padding-inline-start: 24px; }
.answer li { margin: 4px 0; }
.answer li::marker { color: var(--brand-text); font-weight: 600; }
.answer h1, .answer h2, .answer h3, .answer h4 { font-size: 15.5px; font-weight: 650; margin: 12px 0 6px 0; }
.answer strong { font-weight: 650; }
.answer a { color: var(--brand-text); text-decoration: underline; text-underline-offset: 2px; }
.answer code { font-family: var(--mono); font-size: 13px; background: var(--surface-3); padding: 1px 6px; border-radius: 6px; }
.answer pre { background: var(--surface-3); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 12px 14px; overflow-x: auto; }
.answer pre code { background: none; padding: 0; font-size: 13px; }
.answer table { border-collapse: collapse; margin: 8px 0 12px 0; display: block; max-width: 100%; overflow-x: auto; font-size: 14px; }
.answer th, .answer td { border: 1px solid var(--border); padding: 6px 10px; text-align: start; vertical-align: top; }
.answer th { background: var(--surface-2); font-weight: 650; }
.answer blockquote { margin: 8px 0; padding: 4px 14px; border-inline-start: 3px solid var(--brand-soft); color: var(--text-2); }
.answer hr { border: 0; border-top: 1px solid var(--border); margin: 12px 0; }

/* notices: the backend's own "no answer" sentences, labelled */
.bubble.assistant.notice { border-inline-start: 4px solid var(--text-3); background: var(--surface-2); }
.bubble.assistant.notice.empty { border-inline-start-color: var(--text-3); }
.bubble.assistant.notice.outage, .bubble.assistant.notice.limit { border-inline-start-color: var(--warn); background: var(--warn-soft); }
.bubble.assistant.notice.config { border-inline-start-color: var(--warn); background: var(--warn-soft); }
.bubble.assistant.notice.error { border-inline-start-color: var(--danger); background: var(--danger-soft); }
.notice-head { display: flex; align-items: center; gap: 8px; font-weight: 650; font-size: 14px; margin-bottom: 6px; }
.notice-head svg { width: 18px; height: 18px; flex: none; }
.notice-text { margin: 0; }
.notice-hint { margin: 8px 0 0 0; font-size: 13.5px; color: var(--text-2); }

/* processing state */
.typing { display: flex; align-items: center; gap: 10px; color: var(--text-2); font-size: 14px; }
.typing .dots { display: inline-flex; gap: 4px; }
.typing .dots span { width: 7px; height: 7px; border-radius: 50%; background: var(--brand); opacity: 0.35; animation: pulse 1.2s infinite ease-in-out; }
.typing .dots span:nth-child(2) { animation-delay: 0.15s; }
.typing .dots span:nth-child(3) { animation-delay: 0.3s; }
@keyframes pulse { 0%, 80%, 100% { opacity: 0.25; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-2px); } }

/* sources and evidence */
.sources { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; padding-inline-start: 2px; }
.sources-label { font-size: 11.5px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-3); margin-inline-end: 4px; }
.chip {
    display: inline-flex; align-items: center; gap: 6px; max-width: 100%; height: 28px; padding: 0 10px;
    border-radius: var(--radius-pill); border: 1px solid var(--border); background: var(--surface);
    font-size: 12.5px; font-weight: 500; color: var(--text-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.chip.doc { background: var(--surface-2); color: var(--text); font-weight: 600; }
.chip.page { border-color: var(--brand-soft); background: var(--brand-soft); color: var(--brand-text); }
.chip.section { color: var(--text-2); }
.chip.ocr { border-style: dashed; }
.chip.verified { background: var(--brand-fill); border-color: var(--brand-fill); color: #fff; font-weight: 600; }
.chip.verified svg { width: 14px; height: 14px; }

details.evidence { border: 1px solid var(--border); border-radius: var(--radius-md); background: var(--surface); }
details.evidence summary {
    cursor: pointer; list-style: none; display: flex; align-items: center; gap: 8px;
    padding: 10px 14px; font-size: 13.5px; font-weight: 600; color: var(--text-2); border-radius: var(--radius-md);
}
details.evidence summary::-webkit-details-marker { display: none; }
details.evidence summary svg { width: 16px; height: 16px; transition: transform var(--transition); }
details.evidence[open] summary svg { transform: rotate(90deg); }
details.evidence summary:hover { color: var(--text); background: var(--surface-2); }
.evidence-list { margin: 0; padding: 4px 14px 12px 14px; list-style: none; display: flex; flex-direction: column; gap: 10px; }
.evidence-item { border-top: 1px solid var(--border); padding-top: 10px; }
.ev-meta { font-size: 12.5px; font-weight: 600; color: var(--brand-text); margin-bottom: 4px; }
.ev-meta .ev-doc { color: var(--text-2); }
.ev-text { margin: 0; font-size: 13.5px; color: var(--text-2); line-height: 1.55; padding-inline-start: 10px; border-inline-start: 2px solid var(--border-strong); }

.msg-toolbar { display: flex; gap: 6px; }
.icon-btn {
    display: inline-flex; align-items: center; gap: 6px; height: 32px; padding: 0 10px; border-radius: var(--radius-sm);
    border: 1px solid transparent; background: transparent; color: var(--text-3); font-size: 12.5px; font-weight: 600; cursor: pointer;
}
.icon-btn:hover { background: var(--surface-2); color: var(--text); border-color: var(--border); }
.icon-btn svg { width: 15px; height: 15px; }

/* ================================================================
   FIRST-USE SCREEN
   ================================================================ */
.landing { display: flex; flex-direction: column; gap: 26px; padding-top: 12px; }
.hero { display: flex; flex-direction: column; align-items: center; text-align: center; gap: 12px; }
.hero-logo { width: 72px; height: 72px; border-radius: 18px; background: #fff; border: 1px solid var(--border); display: grid; place-items: center; box-shadow: var(--shadow-sm); }
.hero-logo img { width: 60px; height: 60px; object-fit: contain; }
.hero-org { font-size: 12px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--brand-text); }
.hero h2 { font-size: clamp(22px, 2.6vw, 28px); font-weight: 700; line-height: 1.25; color: var(--text); letter-spacing: -0.01em; }
.hero p { max-width: 560px; color: var(--text-2); font-size: 15px; }

.banner { display: flex; gap: 10px; align-items: flex-start; padding: 12px 14px; border-radius: var(--radius-md); border: 1px solid var(--border); background: var(--surface); font-size: 13.5px; color: var(--text-2); }
.banner svg { width: 18px; height: 18px; flex: none; margin-top: 2px; }
.banner strong { color: var(--text); }
.banner.warn { background: var(--warn-soft); border-color: transparent; color: var(--warn); }
.banner.warn strong { color: inherit; }
.banner.danger { background: var(--danger-soft); border-color: transparent; color: var(--danger); }
.banner.danger strong { color: inherit; }
.banner code { font-family: var(--mono); font-size: 12.5px; }

.landing-h { font-size: 12px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-3); }
.topic-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.topic-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 16px 16px 12px 16px; display: flex; flex-direction: column; gap: 8px; box-shadow: var(--shadow-sm); }
.topic-title { font-size: 15px; font-weight: 650; color: var(--text); display: flex; align-items: center; gap: 8px; }
.topic-title .topic-icon { width: 28px; height: 28px; border-radius: 8px; background: var(--brand-soft); color: var(--brand-text); display: grid; place-items: center; flex: none; }
.topic-title svg { width: 16px; height: 16px; }
.topic-blurb { font-size: 13.5px; color: var(--text-2); }
.example-list { display: flex; flex-direction: column; gap: 4px; margin-top: 4px; }
.example-btn.q-btn {
    width: 100%; min-height: 40px; justify-content: flex-start; text-align: start; padding: 8px 10px;
    border-radius: var(--radius-sm); font-size: 13.5px; font-weight: 500; color: var(--brand-text); background: transparent;
    border: 1px solid transparent;
}
.example-btn.q-btn .q-btn__content { justify-content: flex-start; width: 100%; white-space: normal; text-align: start; gap: 8px; }
.example-btn.q-btn:hover { background: var(--brand-soft); border-color: transparent; }
.example-btn .q-icon { font-size: 16px; color: var(--brand-text); }
.example-btn .q-btn__content { flex-wrap: nowrap; align-items: flex-start; }
.example-btn .q-btn__content .q-icon { margin-top: 2px; }
.example-btn .q-btn__content .block { flex: 1 1 auto; min-width: 0; white-space: normal; text-align: start; line-height: 1.45; }

.landing-notes { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; padding: 0; list-style: none; }
.landing-notes li { display: flex; gap: 10px; align-items: flex-start; font-size: 13px; color: var(--text-2); background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 10px 12px; }
.landing-notes svg { width: 16px; height: 16px; flex: none; margin-top: 2px; color: var(--brand-text); }

/* ================================================================
   COMPOSER
   ================================================================ */
.composer-wrap { flex: none; padding: 12px 24px 16px 24px; background: linear-gradient(to bottom, transparent, var(--bg) 30%); }
.composer {
    max-width: var(--content-w); margin: 0 auto; display: flex; align-items: flex-end; gap: 8px;
    padding: 8px 8px 8px 14px; background: var(--surface); border: 1px solid var(--border-strong); border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md); transition: border-color var(--transition), box-shadow var(--transition);
}
.composer:focus-within { border-color: var(--brand); box-shadow: 0 0 0 3px var(--brand-ring), var(--shadow-md); }
.composer.busy { border-color: var(--border); }
.composer .q-field { flex: 1; }
.composer .q-field__control { padding: 0 !important; min-height: 40px; }
.composer .q-field__control::before, .composer .q-field__control::after { display: none; }
.composer .q-field__native { padding: 9px 0; font: inherit; font-size: 15px; line-height: 1.5; color: var(--text); max-height: 200px; overflow-y: auto !important; }
.composer .q-field__native::placeholder { color: var(--text-3); opacity: 1; }
.composer textarea { resize: none; }
.send-btn.q-btn {
    height: 40px; min-height: 40px; padding: 0 14px; border-radius: var(--radius-md);
    background: var(--brand-fill) !important; color: #fff !important; font-weight: 600; font-size: 14px; flex: none;
}
.send-btn.q-btn:hover { background: var(--brand-fill-strong) !important; }
.send-btn.q-btn.disabled, .send-btn.q-btn[aria-disabled="true"] { opacity: 0.55; }
.send-btn .q-btn__content, .send-btn svg, .send-btn .q-icon { color: #fff !important; }
.send-btn .q-icon { font-size: 20px; }
.composer-foot { max-width: var(--content-w); margin: 6px auto 0 auto; display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 0 6px; min-height: 18px; }
.composer-hint { font-size: 12px; color: var(--text-3); }
.composer-hint kbd { font-family: var(--mono); font-size: 11px; padding: 1px 6px; border: 1px solid var(--border-strong); border-bottom-width: 2px; border-radius: 5px; background: var(--surface); color: var(--text-2); }
.composer-help { font-size: 12.5px; color: var(--warn); font-weight: 600; min-height: 18px; }
.composer-help:empty { display: none; }

/* ================================================================
   VOICE INPUT (post-freeze PF-12)
   ================================================================ */
.mic-btn.q-btn { width: 40px; height: 40px; min-height: 40px; flex: none; border-radius: 50%; color: var(--text-2); border: 1px solid transparent; transition: background var(--transition), color var(--transition), box-shadow var(--transition); }
.mic-btn.q-btn .q-icon { font-size: 22px; }
.mic-btn.q-btn:hover { background: var(--surface-2); color: var(--brand-text); }
.mic-btn.voice-listening.q-btn, .mic-btn.voice-starting.q-btn { background: var(--danger-soft); color: var(--danger); border-color: var(--danger); box-shadow: 0 0 0 4px rgba(179, 38, 30, 0.14); animation: micPulse 1.4s ease-in-out infinite; }
.mic-btn.voice-ready.q-btn { color: var(--brand-text); background: var(--brand-soft); }
.mic-btn.voice-error.q-btn { color: var(--warn); background: var(--warn-soft); }
.mic-btn.voice-unsupported.q-btn { color: var(--text-3); background: var(--surface-3); cursor: not-allowed; }
@keyframes micPulse { 0%, 100% { box-shadow: 0 0 0 4px rgba(179, 38, 30, 0.14); } 50% { box-shadow: 0 0 0 9px rgba(179, 38, 30, 0.05); } }
.voice-cancel-btn.q-btn { width: 36px; height: 36px; min-height: 36px; flex: none; border-radius: 50%; color: var(--text-2); }
.voice-cancel-btn.q-btn:hover { background: var(--surface-2); color: var(--danger); }
.voice-cancel-btn.hidden, .voice-strip.hidden { display: none !important; }
.voice-strip { max-width: var(--content-w); margin: 6px auto 0 auto; padding: 0 6px; }
.voice-live { display: flex; align-items: center; gap: 10px; min-height: 28px; padding: 6px 12px; border-radius: var(--radius-md); background: var(--danger-soft); color: var(--text); font-size: 13.5px; }
.voice-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--danger); flex: none; animation: voiceDot 1.2s ease-in-out infinite; }
@keyframes voiceDot { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.35; transform: scale(0.75); } }
.voice-live-label { font-weight: 650; color: var(--danger); flex: none; }
.voice-preview { color: var(--text-2); overflow-wrap: anywhere; min-width: 0; }
.voice-privacy { max-width: var(--content-w); margin: 6px auto 0 auto; padding: 6px 10px; display: flex; gap: 8px; align-items: flex-start; font-size: 12px; color: var(--text-3); line-height: 1.5; }
.voice-privacy svg { width: 14px; height: 14px; flex: none; margin-top: 2px; }
.voice-lang { margin-inline-start: 8px; }
@media (max-width: 540px) {
    .mic-btn.q-btn { width: 38px; height: 38px; min-height: 38px; }
    .voice-lang .lbl { display: none; }
}

/* ================================================================
   DEVELOPER PANEL (reachable through ?dev=1 only)
   ================================================================ */
.dev-panel { border: 1px dashed var(--border-strong); border-radius: var(--radius-md); background: var(--surface-2); font-size: 12.5px; }
.dev-panel summary { cursor: pointer; padding: 8px 12px; font-weight: 600; color: var(--text-2); }
.dev-body { padding: 4px 12px 12px 12px; }
.dev-section { margin: 10px 0 4px 0; font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-3); }
.dev-row { display: flex; justify-content: space-between; gap: 14px; padding: 3px 0; border-bottom: 1px dotted var(--border); }
.dev-row .k { color: var(--text-3); }
.dev-row .v { font-family: var(--mono); font-size: 12px; color: var(--text); text-align: end; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 60%; }

/* ================================================================
   SAVED CONVERSATIONS, DIALOGS, SETTINGS (PF-13)
   ================================================================ */
.session-row { display: flex; align-items: center; gap: 2px; min-width: 0; }
.session-row .session-item.q-btn { flex: 1; min-width: 0; }
.session-menu-btn.q-btn { flex: none; width: 30px; height: 30px; min-height: 30px; padding: 0; color: var(--text-3); opacity: 0; transition: opacity .15s; }
.session-row:hover .session-menu-btn.q-btn, .session-row:focus-within .session-menu-btn.q-btn, .session-row.active .session-menu-btn.q-btn,
.session-menu-btn.q-btn:focus-visible, .history-dialog .session-menu-btn.q-btn { opacity: 1; }
.session-menu-btn.q-btn .q-icon { font-size: 18px; }
.sb-empty { font-size: 12.5px; color: var(--text-3); padding: 6px; font-style: italic; }
.saved-note { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-3); margin-top: 6px; line-height: 1.4; }
.saved-note svg { width: 14px; height: 14px; flex: none; }
.saved-note.warn { color: #9a5b00; }
[data-theme="dark"] .saved-note.warn { color: #f0b35a; }
.dialog-host { display: contents; }
.fob-dialog.q-card { width: min(460px, 92vw); max-width: 92vw; padding: 20px 22px; border-radius: var(--radius-md); background: var(--surface); color: var(--text); align-items: stretch; gap: 0; }
/* Quasar follows the system colour scheme for its fields; the dialog's fields follow the page's own tokens instead. */
.fob-dialog .q-field__native, .fob-dialog .q-field__input, .fob-dialog .q-field__label, .fob-dialog .q-field__marginal,
.fob-dialog .q-select__dropdown-icon, .fob-dialog .q-field__append { color: var(--text) !important; }
.fob-dialog .q-field__native::placeholder { color: var(--text-3) !important; opacity: 1; }
.fob-dialog .q-field--outlined .q-field__control::before { border-color: var(--border-strong) !important; }
.fob-dialog .q-field--outlined .q-field__control:hover::before { border-color: var(--text-3) !important; }
.fob-dialog .q-field--outlined.q-field--focused .q-field__control::after { border-color: var(--brand-fill) !important; }
.fob-dialog .dialog-title { font-size: 18px; font-weight: 700; margin: 0 0 4px 0; line-height: 1.3; }
.fob-dialog .dialog-sub, .fob-dialog .settings-note { font-size: 13px; color: var(--text-2); line-height: 1.5; margin: 0 0 10px 0; }
.fob-dialog .dialog-text { font-size: 14px; line-height: 1.55; margin: 8px 0 6px 0; overflow-wrap: anywhere; }
.dialog-actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
.dialog-actions.spread { justify-content: space-between; }
.settings-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 8px 0; border-bottom: 1px solid var(--border); }
.settings-row .settings-lbl { font-size: 14px; font-weight: 500; }
.settings-row .q-select { min-width: 170px; }
.settings-section { font-size: 11.5px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--text-3); margin: 14px 0 4px 0; }
.history-dialog .session-list { max-height: 60vh; overflow: auto; margin-top: 8px; }
.topbar-history { display: none; }
.q-menu, .q-dialog .q-card { background: var(--surface); color: var(--text); }
.q-menu .q-item { color: var(--text); min-height: 40px; }
.q-menu .q-item:hover { background: var(--surface-2); }
.q-dialog__inner > .q-card.fob-dialog { box-shadow: 0 18px 50px rgba(0,0,0,.28); }
@media (max-width: 900px) { .topbar-history { display: flex; } }
@media (max-width: 540px) {
    .fob-dialog.q-card { padding: 16px; }
    .settings-row { flex-direction: column; align-items: stretch; gap: 4px; }
    .settings-row .q-select { min-width: 0; width: 100%; }
    .dialog-actions.spread { flex-direction: column-reverse; align-items: stretch; }
}

/* ================================================================
   ONBOARDING PAGES (PF-14): shell, dashboard, procedure, finder
   ================================================================ */
.ob-root { min-height: 100vh; display: flex; flex-direction: column; background: var(--bg); color: var(--text); }
/* Quasar follows the system colour scheme (body--dark paints text white); the pages' text follows the page tokens instead. */
.ob-root h1, .ob-root h2, .ob-root h3, .ob-root p, .ob-root dd, .ob-root dt, .ob-root li, .ob-root blockquote, .ob-root q, .ob-root label, .ob-root summary { color: inherit; }
.ob-root .ob-h1, .ob-root .ob-h2, .ob-root .ob-task-title, .ob-root .rf-title, .ob-root .ob-stat dd, .ob-root .pr-step-text, .ob-root .pr-conditions, .ob-root .pr-task-box, .ob-root .ob-summary, .ob-root .ob-task, .ob-root .rf-card, .ob-root .pr-card { color: var(--text); }
.ob-topbar { position: sticky; top: 0; z-index: 20; display: flex; align-items: center; gap: 16px; height: 60px; padding: 0 20px; background: var(--surface); border-bottom: 1px solid var(--border); }
.ob-brand { display: flex; align-items: center; gap: 10px; text-decoration: none; color: var(--text); min-width: 0; }
.ob-brand img { width: 36px; height: 36px; border-radius: 8px; object-fit: cover; }
.ob-brand span { display: flex; flex-direction: column; line-height: 1.15; }
.ob-brand .brand-org { font-size: 10.5px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--brand-text); }
.ob-brand .brand-name { font-size: 14px; font-weight: 700; }
.ob-nav { display: flex; gap: 4px; margin-inline-start: 8px; flex: 1; min-width: 0; }
.ob-nav-link { display: inline-flex; align-items: center; gap: 6px; padding: 8px 12px; border-radius: var(--radius-pill); font-size: 13.5px; font-weight: 600; color: var(--text-2); text-decoration: none; white-space: nowrap; }
.ob-nav-link svg { width: 16px; height: 16px; }
.ob-nav-link:hover { background: var(--surface-2); color: var(--text); }
.ob-nav-link.active { background: var(--brand-soft); color: var(--brand-text); }
.ob-nav-link:focus-visible { outline: 2px solid var(--brand); outline-offset: 2px; }
.ob-topbar-right { display: flex; align-items: center; gap: 6px; }
.ob-main { width: 100%; max-width: 1040px; margin: 0 auto; padding: 24px 20px 48px 20px; box-sizing: border-box; }
.ob-live:empty { display: none; }
.page-note { display: flex; align-items: center; gap: 8px; padding: 10px 14px; border-radius: var(--radius-md); background: var(--info-soft); color: var(--text); font-size: 14px; margin-bottom: 14px; }
.page-note.ok { background: var(--brand-soft); color: var(--brand-text); }
.page-note.warn { background: var(--warn-soft); color: var(--warn); }
.page-note svg { width: 16px; height: 16px; flex: none; }
.ob-page-head { margin-bottom: 18px; }
.ob-h1 { font-size: 26px; font-weight: 800; letter-spacing: -0.01em; margin: 0 0 6px 0; line-height: 1.25; }
.ob-h2 { font-size: 18px; font-weight: 700; margin: 0 0 8px 0; }
.ob-h3 { font-size: 14px; font-weight: 700; margin: 12px 0 6px 0; color: var(--text-2); }
.ob-sub { font-size: 14.5px; color: var(--text-2); line-height: 1.55; margin: 0; max-width: 72ch; }
.ob-link { color: var(--brand-text); font-weight: 600; text-decoration: none; }
.ob-link:hover { text-decoration: underline; }
.ob-note { font-size: 13px; color: var(--text-2); margin: 6px 0 0 0; }
.ob-note.ok { color: var(--brand-text); display: flex; gap: 6px; align-items: center; }
.ob-note.ok svg { width: 16px; height: 16px; }
.ob-foot { font-size: 12.5px; color: var(--text-3); line-height: 1.55; margin-top: 28px; max-width: 80ch; }
.ob-empty { display: flex; align-items: center; gap: 10px; padding: 18px; border: 1px dashed var(--border-strong); border-radius: var(--radius-md); color: var(--text-2); font-size: 14px; flex-wrap: wrap; }
.ob-empty svg { width: 18px; height: 18px; flex: none; }
.banner.warn { margin-bottom: 14px; }
/* summary */
.ob-summary { display: flex; gap: 22px; align-items: flex-start; padding: 20px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); box-shadow: var(--shadow-sm); margin-bottom: 18px; }
.ob-ring { --pct: 0; width: 112px; height: 112px; flex: none; border-radius: 50%; display: grid; place-items: center;
    background: conic-gradient(var(--brand) calc(var(--pct) * 1%), var(--surface-3) 0); }
.ob-ring::before { content: ""; position: absolute; width: 86px; height: 86px; border-radius: 50%; background: var(--surface); }
.ob-ring { position: relative; }
.ob-ring-value { position: relative; font-size: 22px; font-weight: 800; color: var(--brand-text); }
.ob-summary-body { flex: 1; min-width: 0; }
.ob-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 0 0 10px 0; }
.ob-stat { background: var(--surface-2); border-radius: var(--radius-sm); padding: 8px 10px; }
.ob-stat dt { font-size: 11.5px; color: var(--text-3); font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
.ob-stat dd { margin: 2px 0 0 0; font-size: 20px; font-weight: 800; }
.ob-cat-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip.cat { display: inline-flex; align-items: center; gap: 5px; }
.chip.cat svg { width: 14px; height: 14px; }
.ob-next { margin: 0; padding-inline-start: 18px; font-size: 14px; line-height: 1.7; }
.ob-next a { color: var(--text); font-weight: 600; text-decoration: none; }
.ob-next a:hover { text-decoration: underline; }
/* filters */
.ob-filters { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 14px 0; }
.ob-filter.q-btn { border-radius: var(--radius-pill); border: 1px solid var(--border); color: var(--text-2); padding: 4px 14px; font-weight: 600; }
.ob-filter.active.q-btn { background: var(--brand-soft); border-color: transparent; color: var(--brand-text); }
/* task cards */
.ob-tasks { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; }
.ob-task { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 16px 16px 14px 16px; box-shadow: var(--shadow-sm); display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.ob-task.done { border-color: var(--brand-soft); background: linear-gradient(180deg, var(--brand-soft) 0, var(--surface) 72px); }
.ob-task.highlight { outline: 2px solid var(--brand); outline-offset: 2px; }
.ob-task-head { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; font-size: 12px; }
.ob-task-cat { display: inline-flex; align-items: center; gap: 5px; color: var(--text-3); font-weight: 600; }
.ob-task-cat svg { width: 14px; height: 14px; }
.kind { display: inline-block; padding: 2px 8px; border-radius: var(--radius-pill); font-size: 11.5px; font-weight: 700; background: var(--surface-3); color: var(--text-2); }
.kind.required { background: var(--danger-soft); color: var(--danger); }
.kind.recommended { background: var(--brand-soft); color: var(--brand-text); }
.kind.optional { background: var(--info-soft); color: var(--text-2); }
.ob-status-pill { margin-inline-start: auto; padding: 2px 8px; border-radius: var(--radius-pill); font-size: 11.5px; font-weight: 700; background: var(--surface-3); color: var(--text-2); }
.ob-status-pill.completed { background: var(--brand-fill); color: #fff; }
.ob-status-pill.in_progress { background: var(--warn-soft); color: var(--warn); }
.ob-status-pill.not_applicable { background: var(--surface-3); color: var(--text-3); text-decoration: line-through; }
.ob-task-title { font-size: 16px; font-weight: 700; margin: 0; line-height: 1.35; }
.ob-task-desc { margin: 0; font-size: 13.5px; color: var(--text-2); line-height: 1.55; }
.ob-task-timing { margin: 0; font-size: 12.5px; color: var(--text-3); display: flex; gap: 6px; align-items: center; }
.ob-task-timing svg { width: 14px; height: 14px; flex: none; }
.ob-evidence { font-size: 13px; }
.ob-evidence summary { cursor: pointer; color: var(--brand-text); font-weight: 600; display: flex; align-items: center; gap: 6px; list-style: none; }
.ob-evidence summary::-webkit-details-marker { display: none; }
.ob-evidence summary svg { width: 14px; height: 14px; }
.ob-ev-count { background: var(--surface-3); color: var(--text-2); border-radius: var(--radius-pill); padding: 0 7px; font-size: 11px; }
.ob-evidence ul { margin: 8px 0 0 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: 8px; }
.ob-ev-src { display: block; font-size: 12px; color: var(--text-3); font-weight: 600; margin-bottom: 2px; }
.ob-evidence blockquote, .pr-step-text { margin: 0; padding: 8px 12px; border-inline-start: 3px solid var(--brand); background: var(--surface-2); border-radius: 6px; font-size: 13.5px; line-height: 1.55; color: var(--text); overflow-wrap: anywhere; }
.ob-task-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.ob-btn.q-btn { border-radius: var(--radius-pill); font-weight: 600; font-size: 13px; }
.ob-btn.primary.q-btn { background: var(--brand-fill) !important; color: #fff !important; }
.ob-btn.complete.q-btn { background: var(--brand-fill) !important; color: #fff !important; }
.ob-btn.small.q-btn { font-size: 12.5px; }
.ob-task-status { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 6px; padding-top: 10px; border-top: 1px solid var(--border); }
.ob-task-status .lbl { font-size: 12.5px; color: var(--text-3); font-weight: 600; }
.ob-status-select { min-width: 170px; }
.ob-root .q-field__native, .ob-root .q-field__input, .ob-root .q-field__label, .ob-root .q-select__dropdown-icon, .ob-root .q-field__append, .ob-root .q-field__marginal { color: var(--text) !important; }
.ob-root .q-field--outlined .q-field__control::before { border-color: var(--border-strong) !important; }
.ob-root .q-field--outlined.q-field--focused .q-field__control::after { border-color: var(--brand-fill) !important; }
.ob-root .q-field__native::placeholder { color: var(--text-3) !important; opacity: 1; }
/* PF-16 (D-127): the field counter and hint follow the page tokens too (Quasar painted them in the OS colour scheme:
   white on white in the light theme when the OS is dark, measured contrast 1:1). An error message keeps Quasar's colour. */
.ob-root .q-field__counter, .ob-root .q-field:not(.q-field--error) .q-field__messages { color: var(--text-3) !important; }
/* procedure */
.ob-crumbs { font-size: 13px; color: var(--text-3); margin-bottom: 10px; display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.pr-heading { margin: 8px 0 0 0; font-size: 13.5px; color: var(--text-2); }
.pr-heading q { font-weight: 600; color: var(--text); }
.pr-conditions { display: flex; gap: 8px; align-items: flex-start; margin: 10px 0 0 0; padding: 10px 12px; background: var(--info-soft); border-radius: var(--radius-md); font-size: 13.5px; line-height: 1.5; max-width: 80ch; }
.pr-conditions svg { width: 16px; height: 16px; flex: none; margin-top: 2px; }
.pr-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 20px; box-shadow: var(--shadow-sm); display: flex; flex-direction: column; gap: 12px; }
.pr-progress { height: 6px; background: var(--surface-3); border-radius: 3px; overflow: hidden; }
.pr-progress span { display: block; height: 100%; background: var(--brand); transition: width var(--transition); }
.pr-step-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.pr-counter { font-size: 13px; font-weight: 700; color: var(--brand-text); text-transform: uppercase; letter-spacing: .06em; }
.pr-flag { display: inline-flex; align-items: center; gap: 5px; padding: 2px 10px; border-radius: var(--radius-pill); font-size: 12px; font-weight: 700; }
.pr-flag svg { width: 14px; height: 14px; }
.pr-flag.conditional { background: var(--info-soft); color: var(--text-2); }
.pr-flag.after { background: var(--surface-3); color: var(--text-2); }
.pr-step-text { font-size: 17px; line-height: 1.6; padding: 14px 18px; }
.pr-quote-note { margin: 0; font-size: 12px; color: var(--text-3); }
.pr-controls { display: flex; flex-wrap: wrap; gap: 8px; }
.pr-evidence { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding-top: 8px; }
.pr-dots { display: flex; flex-wrap: wrap; gap: 6px; padding-top: 4px; }
.pr-dot { width: 14px; height: 14px; border-radius: 50%; border: 2px solid var(--border-strong); background: var(--surface); cursor: pointer; padding: 0; }
.pr-dot.current { background: var(--brand); border-color: var(--brand); }
.pr-dot.conditional { border-style: dashed; }
.pr-dot.after { border-color: var(--text-3); }
.pr-dot:focus-visible { outline: 2px solid var(--brand); outline-offset: 2px; }
.pr-task-box { margin-top: 16px; padding: 16px 18px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); }
/* finder */
.rf-context-box { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; padding: 10px 14px; margin-bottom: 12px; background: var(--brand-soft); color: var(--brand-text); border-radius: var(--radius-md); font-size: 14px; }
.rf-search { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 8px; }
.rf-input { flex: 1; min-width: 240px; }
.rf-count { font-size: 13px; color: var(--text-3); font-weight: 600; }
.rf-results { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; }
.rf-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 16px; box-shadow: var(--shadow-sm); display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.rf-card.contextual { border-color: var(--brand); }
.rf-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: 12px; }
.rf-type { padding: 2px 8px; border-radius: var(--radius-pill); font-weight: 700; font-size: 11.5px; background: var(--surface-3); color: var(--text-2); }
.rf-type.document { background: var(--brand-soft); color: var(--brand-text); }
.rf-type.website { background: var(--info-soft); }
.rf-context { margin-inline-start: auto; color: var(--brand-text); font-weight: 700; }
.rf-title { margin: 0; font-size: 16px; font-weight: 700; line-height: 1.35; }
.rf-desc { margin: 0; font-size: 13.5px; color: var(--text-2); line-height: 1.55; }
.rf-meta { display: flex; flex-wrap: wrap; gap: 6px; }
.rf-contacts { background: var(--surface-2); border-radius: var(--radius-sm); padding: 8px 12px; font-size: 13px; }
.rf-contacts-head { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-3); font-weight: 600; margin-bottom: 4px; }
.rf-contacts-head svg { width: 14px; height: 14px; }
.rf-contacts ul { margin: 0; padding: 0; list-style: none; }
.rf-contact-label { color: var(--text-2); }
.rf-contact-value { font-weight: 700; font-variant-numeric: tabular-nums; unicode-bidi: isolate; }
.rf-contact-note { display: flex; align-items: center; gap: 6px; margin: 0; font-size: 12.5px; color: var(--text-3); }
.rf-contact-note svg { width: 14px; height: 14px; flex: none; }
.rf-link { font-size: 13.5px; line-height: 1.5; }
.rf-link svg { width: 14px; height: 14px; vertical-align: -2px; margin-inline-end: 4px; }
.res-link { color: var(--brand-text); font-weight: 600; word-break: break-all; }
.rf-link-note { color: var(--text-3); font-size: 12px; }
.rf-link-desc { color: var(--text-2); font-size: 12.5px; }
.rf-related { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; font-size: 12.5px; }
.rf-related .lbl { color: var(--text-3); font-weight: 600; }
.chip.task { text-decoration: none; cursor: pointer; }
.chip.task:hover { background: var(--brand-soft); color: var(--brand-text); }
/* chat page additions */
.sb-nav { display: flex; flex-direction: column; gap: 4px; margin: 4px 0 8px 0; }
.sb-nav a { display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: var(--radius-sm); font-size: 13.5px; font-weight: 600; color: var(--text-2); text-decoration: none; }
.sb-nav a svg { width: 16px; height: 16px; }
.sb-nav a:hover { background: var(--surface-2); color: var(--text); }
.chat-context { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 0 0 8px 0; padding: 8px 12px; background: var(--brand-soft); color: var(--brand-text); border-radius: var(--radius-md); font-size: 13px; }
.chat-context a { color: inherit; font-weight: 700; }
.chat-context svg { width: 16px; height: 16px; flex: none; }
.chat-context-wrap:empty { display: none; }
@media (max-width: 900px) {
    .ob-topbar { padding: 0 12px; gap: 10px; }
    .ob-brand span { display: none; }
    .ob-nav-link { padding: 8px 10px; font-size: 13px; }
    .ob-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 540px) {
    .ob-topbar { flex-wrap: wrap; height: auto; padding: 8px 10px; row-gap: 4px; }
    .ob-nav { order: 3; flex: 0 0 100%; width: 100%; justify-content: center; margin: 0; }
    .ob-topbar-right { margin-inline-start: auto; }
    .ob-main { padding: 16px 12px 40px 12px; }
    .ob-h1 { font-size: 22px; }
    .ob-nav-link span, .ob-nav-link { font-size: 12px; }
    .ob-nav-link { padding: 6px 8px; }
    .ob-summary { flex-direction: column; align-items: center; text-align: center; }
    .ob-cat-chips { justify-content: center; }
    .ob-tasks, .rf-results { grid-template-columns: 1fr; }
    .ob-status-select { min-width: 0; width: 100%; }
    .pr-step-text { font-size: 15.5px; }
    .pr-controls .ob-btn { flex: 1 1 auto; }
}

/* ================================================================
   PF-15: PLATFORM SHELL, NAVIGATION AND PAGES
   ================================================================ */
/* Quasar's "primary" follows the brand everywhere (NiceGUI's default is a light blue) */
:root { --q-primary: #0B7A52; --q-negative: #B3261E; }
[data-theme="dark"] { --q-primary: #2EA97C; --q-negative: #F28B82; }
body .text-primary { color: var(--brand-text) !important; }
body .bg-primary { background: var(--brand-fill) !important; color: #fff !important; }
body .text-negative { color: var(--danger) !important; }
body .bg-negative { background: #B3261E !important; color: #fff !important; }
.topbar-btn.q-btn.text-primary, .topbar-btn.q-btn .q-icon { color: var(--text-2) !important; }
.topbar-btn.q-btn:hover .q-icon { color: var(--brand-text) !important; }
.q-btn:not(.q-btn--round):active { transform: translateY(1px); }

/* ---- sidebar: brand, primary navigation, guest chip ---- */
.sidebar { gap: 10px; }
.sidebar .new-chat-btn.q-btn { margin-top: 4px; }
.nav-block { display: flex; flex-direction: column; gap: 2px; }
.nav-section { font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--text-3); padding: 10px 12px 4px 12px; }
.nav-item { position: relative; display: flex; align-items: center; gap: 11px; min-height: 40px; padding: 8px 12px; border-radius: var(--radius-sm);
    color: var(--text-2); font-size: 14px; font-weight: 550; text-decoration: none; transition: background var(--transition), color var(--transition); }
.nav-item svg { width: 18px; height: 18px; flex: none; color: var(--text-3); transition: color var(--transition); }
.nav-item:hover { background: var(--surface-2); color: var(--text); }
.nav-item:hover svg { color: var(--brand-text); }
.nav-item.active { background: var(--brand-soft); color: var(--brand-text); font-weight: 650; }
.nav-item.active svg { color: var(--brand-text); }
.nav-item.active::before { content: ""; position: absolute; inset-inline-start: -16px; top: 8px; bottom: 8px; width: 3px; border-radius: 0 3px 3px 0; background: var(--brand); }
html[dir="rtl"] .nav-item.active::before { border-radius: 3px 0 0 3px; }
.nav-item .nav-count { margin-inline-start: auto; font-size: 11.5px; font-weight: 700; color: var(--text-3); background: var(--surface-3); border-radius: var(--radius-pill); padding: 0 8px; }
.sb-recent-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 10px 12px 2px 12px; }
.sb-recent-head .nav-section { padding: 0; }
.sb-view-all { font-size: 12.5px; font-weight: 600; color: var(--brand-text); text-decoration: none; white-space: nowrap; }
.sb-view-all:hover { text-decoration: underline; }
.guest-chip { display: flex; align-items: center; gap: 10px; padding: 8px 10px; border-radius: var(--radius-md); text-decoration: none; color: var(--text); border: 1px solid var(--border); background: var(--surface-2); transition: border-color var(--transition), background var(--transition); }
.guest-chip:hover { border-color: var(--border-strong); background: var(--surface-3); }
.guest-avatar { width: 34px; height: 34px; flex: none; border-radius: 50%; display: grid; place-items: center; background: var(--brand-fill); color: #fff; font-weight: 700; font-size: 14px; }
.guest-avatar.large { width: 64px; height: 64px; font-size: 26px; }
.guest-text { display: flex; flex-direction: column; min-width: 0; line-height: 1.25; }
.guest-name { font-size: 13.5px; font-weight: 650; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.guest-sub { font-size: 11.5px; color: var(--text-3); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
details.about { font-size: 12.5px; color: var(--text-2); line-height: 1.55; padding: 0 4px; }
details.about summary { cursor: pointer; font-weight: 650; color: var(--text-2); list-style: none; padding: 4px 0; }
details.about summary::-webkit-details-marker { display: none; }
details.about[open] summary { color: var(--text); }
.drawer-close { display: none; width: 40px; height: 40px; align-items: center; justify-content: center; border-radius: var(--radius-sm); border: 1px solid var(--border);
    background: var(--surface); color: var(--text-2); cursor: pointer; padding: 0; z-index: 2; }
.drawer-close svg { width: 18px; height: 18px; }
.sidebar { position: relative; }
.brand { color: var(--text); }
.new-chat-link { display: flex; align-items: center; justify-content: center; gap: 8px; min-height: 44px; border-radius: var(--radius-md); background: var(--brand-fill); color: #fff;
    font-weight: 600; font-size: 14px; text-decoration: none; box-shadow: var(--shadow-sm); margin-top: 4px; transition: background var(--transition); }
.new-chat-link:hover { background: var(--brand-fill-strong); }
.new-chat-link svg { width: 18px; height: 18px; }
.link-btn { display: inline-flex; align-items: center; gap: 6px; min-height: 36px; padding: 6px 16px; border-radius: var(--radius-pill); background: var(--brand-fill); color: #fff;
    font-weight: 650; font-size: 13.5px; text-decoration: none; transition: background var(--transition); }
.link-btn:hover { background: var(--brand-fill-strong); }
.link-btn.secondary { background: transparent; color: var(--brand-text); border: 1px solid var(--border-strong); }
.link-btn.secondary:hover { background: var(--surface-2); }
.link-btn svg { width: 16px; height: 16px; }
.saved-time { white-space: nowrap; }
.rf-contact-value .tel { white-space: nowrap; display: inline-block; }
.answer .tel, .original-body .tel { white-space: nowrap; unicode-bidi: isolate; font-variant-numeric: tabular-nums; }
.composer-help:empty, .composer-foot .nicegui-html.composer-help:empty { display: none; }
.sb-recent-head { flex-wrap: nowrap; }
.sb-recent-head .nav-section { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* conversation titles never run under the menu or out of the sidebar (audit finding 2) */
.session-row { min-width: 0; width: 100%; }
.session-item.q-btn { overflow: hidden; }
.session-item.q-btn .q-btn__content { min-width: 0; flex-wrap: nowrap; }
.session-item .nicegui-html, .session-item .q-btn__content > div { min-width: 0; max-width: 100%; overflow: hidden; flex: 1 1 auto; }
.session-item .session-title { max-width: 100%; }
.session-menu-btn.q-btn { opacity: .55; }

/* ---- header: menu button and page title ---- */
.nav-toggle { display: none; width: 40px; height: 40px; flex: none; align-items: center; justify-content: center; border-radius: var(--radius-sm);
    border: 1px solid var(--border); background: var(--surface); color: var(--text-2); cursor: pointer; padding: 0; }
.nav-toggle svg { width: 20px; height: 20px; }
.nav-toggle:hover { color: var(--brand-text); border-color: var(--border-strong); }
.nav-backdrop { display: none; }
.page-topbar .topbar-title { font-size: 15px; }
.topbar-crumb { font-size: 12.5px; color: var(--text-3); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* ---- page frame for every page other than the chat ---- */
.page-scroll { flex: 1; min-height: 0; overflow-y: auto; overflow-x: hidden; }
.page-main { width: 100%; max-width: 1080px; margin: 0 auto; padding: 28px 28px 56px 28px; box-sizing: border-box; }
.page-head { margin-bottom: 20px; display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.page-head-text { min-width: 0; flex: 1 1 420px; }
.page-title { font-size: 26px; font-weight: 800; letter-spacing: -0.01em; margin: 0 0 6px 0; line-height: 1.25; color: var(--text); }
.page-sub { font-size: 14.5px; color: var(--text-2); line-height: 1.55; margin: 0; max-width: 72ch; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 20px 22px; box-shadow: var(--shadow-sm); color: var(--text); min-width: 0; }
.card + .card { margin-top: 16px; }
.card-head { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 4px; }
.card-icon { width: 36px; height: 36px; flex: none; border-radius: var(--radius-sm); display: grid; place-items: center; background: var(--brand-soft); color: var(--brand-text); }
.card-icon svg { width: 18px; height: 18px; }
.card-title { font-size: 17px; font-weight: 700; margin: 0; color: var(--text); }
.card-desc { font-size: 13.5px; color: var(--text-2); margin: 2px 0 0 0; line-height: 1.5; }
.grid-2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; align-items: start; }
.grid-2 > .card + .card { margin-top: 0; }
.muted { color: var(--text-3); }
.quiet-link { color: var(--brand-text); font-weight: 600; text-decoration: none; display: inline-flex; align-items: center; gap: 6px; }
.quiet-link:hover { text-decoration: underline; }
.quiet-link svg { width: 15px; height: 15px; }
html[dir="rtl"] .quiet-link svg.arrow-icon { transform: scaleX(-1); }
.btn-primary.q-btn { background: var(--brand-fill) !important; color: #fff !important; border-radius: var(--radius-pill); font-weight: 650; padding: 4px 16px; min-height: 38px; }
.btn-secondary.q-btn { border-radius: var(--radius-pill); font-weight: 600; padding: 4px 14px; min-height: 38px; color: var(--brand-text) !important; }
.btn-secondary.q-btn::before { border-color: var(--border-strong) !important; }
.btn-danger.q-btn { border-radius: var(--radius-pill); font-weight: 600; padding: 4px 14px; min-height: 38px; color: var(--danger) !important; }
.btn-danger.q-btn::before { border-color: var(--danger) !important; }
.page-note { animation: noteIn 220ms ease both; }
@keyframes noteIn { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: none; } }
@keyframes drawerIn { from { opacity: 0; transform: translateX(-14px); } to { opacity: 1; transform: none; } }
@keyframes drawerInRtl { from { opacity: 0; transform: translateX(14px); } to { opacity: 1; transform: none; } }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

/* ---- settings page ---- */
.setting-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px 20px; align-items: center; padding: 14px 0; border-top: 1px solid var(--border); }
.setting-row:first-of-type { margin-top: 10px; }
.setting-label { font-size: 14.5px; font-weight: 650; color: var(--text); }
.setting-help { font-size: 12.5px; color: var(--text-3); margin-top: 2px; line-height: 1.45; }
.seg.q-btn-group { border: 1px solid var(--border-strong); border-radius: var(--radius-pill); overflow: hidden; box-shadow: none !important; background: var(--surface); flex-wrap: nowrap; }
.seg.q-btn-group > .q-btn { min-height: 36px; padding: 4px 14px; font-weight: 600; font-size: 13.5px; color: var(--text-2) !important; background: transparent !important; border-radius: 0; }
.seg.q-btn-group > .q-btn + .q-btn { border-inline-start: 1px solid var(--border); }
body .seg.q-btn-group > .q-btn.bg-primary { background: var(--brand-fill) !important; color: #fff !important; }
.data-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; padding: 12px 0; border-top: 1px solid var(--border); font-size: 14px; }
.data-actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }

/* ---- profile page ---- */
.profile-hero { display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
.profile-name { font-size: 22px; font-weight: 800; margin: 0; color: var(--text); overflow-wrap: anywhere; }
.badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: var(--radius-pill); font-size: 12px; font-weight: 700; background: var(--brand-soft); color: var(--brand-text); }
.badge svg { width: 14px; height: 14px; }
.facts { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 6px 16px; margin: 12px 0 0 0; font-size: 14px; }
.facts dt { color: var(--text-3); font-weight: 600; }
.facts dd { margin: 0; color: var(--text); overflow-wrap: anywhere; }
.name-form { display: flex; gap: 10px; align-items: flex-start; flex-wrap: wrap; margin-top: 10px; }
.name-form .q-field { flex: 1 1 240px; }
.progress-bar { height: 10px; border-radius: 5px; background: var(--surface-3); overflow: hidden; margin: 10px 0 8px 0; }
.progress-bar > span { display: block; height: 100%; background: var(--brand); border-radius: 5px; transition: width 400ms ease; }
.activity { list-style: none; margin: 8px 0 0 0; padding: 0; display: flex; flex-direction: column; }
.activity li { display: flex; gap: 10px; align-items: flex-start; padding: 10px 0; border-top: 1px solid var(--border); font-size: 14px; }
.activity li:first-child { border-top: 0; }
.activity .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--brand); flex: none; margin-top: 7px; }
.activity .dot.conv { background: var(--text-3); }
.activity time, .hs-meta time { color: var(--text-3); font-size: 12.5px; white-space: nowrap; }

/* ---- history page ---- */
.toolbar { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 16px; }
.toolbar .q-field { flex: 1 1 260px; }
.hs-group { margin-top: 18px; }
.hs-group-title { font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--text-3); margin: 0 0 8px 4px; }
.hs-list { display: flex; flex-direction: column; gap: 8px; }
.hs-row { display: flex; align-items: center; gap: 12px; padding: 12px 14px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md); transition: border-color var(--transition), box-shadow var(--transition); min-width: 0; }
.hs-row:hover { border-color: var(--border-strong); box-shadow: var(--shadow-sm); }
.hs-row.current { border-color: var(--brand); }
.hs-icon { width: 34px; height: 34px; flex: none; border-radius: var(--radius-sm); display: grid; place-items: center; background: var(--surface-3); color: var(--text-2); }
.hs-icon svg { width: 17px; height: 17px; }
.hs-main { flex: 1; min-width: 0; }
.hs-title.q-btn { padding: 0; min-height: 0; font-size: 15px; font-weight: 650; color: var(--text) !important; max-width: 100%; justify-content: flex-start; }
.hs-title.q-btn .q-btn__content { min-width: 0; justify-content: flex-start; }
.hs-title .nicegui-html, .hs-title .q-btn__content > div { min-width: 0; overflow: hidden; }
.hs-title-text { display: block; max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-align: start; }
.hs-title.q-btn:hover .hs-title-text { text-decoration: underline; color: var(--brand-text); }
.hs-meta { display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: center; font-size: 12.5px; color: var(--text-3); margin-top: 2px; }
.hs-actions { display: flex; gap: 4px; flex: none; }
.icon-action.q-btn { width: 36px; height: 36px; min-height: 36px; padding: 0; color: var(--text-2) !important; border-radius: var(--radius-sm); }
.icon-action.q-btn:hover { background: var(--surface-2); color: var(--brand-text) !important; }
.icon-action.danger.q-btn:hover { color: var(--danger) !important; }
.icon-action svg { width: 17px; height: 17px; }
.empty-state { text-align: center; padding: 40px 20px; border: 1px dashed var(--border-strong); border-radius: var(--radius-lg); background: var(--surface); }
.empty-state .empty-icon { width: 48px; height: 48px; margin: 0 auto 10px auto; border-radius: 50%; display: grid; place-items: center; background: var(--brand-soft); color: var(--brand-text); }
.empty-state .empty-icon svg { width: 22px; height: 22px; }
.empty-state h2 { font-size: 17px; margin: 0 0 4px 0; color: var(--text); }
.empty-state p { color: var(--text-2); font-size: 14px; margin: 0 0 14px 0; }

/* ---- procedures index ---- */
.px-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; }
.px-card { display: flex; flex-direction: column; gap: 10px; transition: border-color var(--transition), box-shadow var(--transition), transform var(--transition); }
.px-card:hover { border-color: var(--border-strong); box-shadow: var(--shadow-md); transform: translateY(-1px); }
.px-meta { display: flex; flex-wrap: wrap; gap: 6px; }
.px-task { font-size: 13px; color: var(--text-2); display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.px-foot { margin-top: auto; display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; padding-top: 8px; border-top: 1px solid var(--border); }
.px-position { font-size: 13px; color: var(--text-3); }

/* ---- chat refinements ---- */
.msg-meta { display: flex; gap: 8px; align-items: center; margin-top: 4px; font-size: 11.5px; color: var(--text-3); }
.msg-row.user .msg-meta { justify-content: flex-end; }
.msg-time { white-space: nowrap; }
.msg-toolbar { display: flex; align-items: center; gap: 12px; }
.msg-toolbar .msg-time { font-size: 11.5px; color: var(--text-3); }
.sources .chip.verified { box-shadow: 0 1px 0 rgba(0,0,0,.04); }
.composer-wrap .chat-context { margin-inline: auto; max-width: var(--content-w); }

/* ---- PF-14 pages inside the shell ---- */
.app-shell .ob-main { max-width: 1080px; }
.ob-task-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; flex-wrap: nowrap; }
.ob-task-tags { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; min-width: 0; }
.ob-task-head .ob-status-pill { margin-inline-start: 0; flex: none; }
.ob-btn.q-btn { padding: 4px 14px; min-height: 34px; }
.ob-btn.q-btn .q-icon { font-size: 18px; }
.ob-btn.q-btn:not(.primary):not(.complete) { color: var(--brand-text) !important; }
.ob-task, .rf-card { transition: border-color var(--transition), box-shadow var(--transition); }
.ob-task:hover, .rf-card:hover { border-color: var(--border-strong); box-shadow: var(--shadow-md); }
.ob-task.just-done { animation: doneFlash 900ms ease 1; }
@keyframes doneFlash { 0% { box-shadow: 0 0 0 0 var(--brand-ring); } 40% { box-shadow: 0 0 0 6px var(--brand-ring); } 100% { box-shadow: var(--shadow-sm); } }
.rf-source { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; font-size: 12.5px; color: var(--text-3); }
.rf-source svg { width: 14px; height: 14px; }
.rf-source .verified { color: var(--brand-text); font-weight: 650; display: inline-flex; align-items: center; gap: 4px; }
.rf-related .chip.task { max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rf-link { display: flex; flex-direction: column; gap: 2px; padding: 8px 12px; border-radius: var(--radius-sm); background: var(--surface-2); }
.rf-link-head { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.rf-counts { font-size: 13px; color: var(--text-3); margin: 0 0 12px 0; }
.rf-contacts .rf-contact-value { font-size: 14px; }

/* ---- accessibility: large text and reduced motion (Settings → Accessibility) ---- */
html[data-text="large"] body { font-size: 17px; }
html[data-text="large"] .bubble, html[data-text="large"] .answer, html[data-text="large"] .answer p, html[data-text="large"] .answer li { font-size: 17.5px; }
html[data-text="large"] .composer .q-field__native { font-size: 17px; }
html[data-text="large"] .nav-item, html[data-text="large"] .session-item.q-btn, html[data-text="large"] .example-btn.q-btn { font-size: 15.5px; }
html[data-text="large"] .ob-task-desc, html[data-text="large"] .rf-desc, html[data-text="large"] .page-sub, html[data-text="large"] .card-desc,
html[data-text="large"] .setting-help, html[data-text="large"] .hs-meta, html[data-text="large"] .ob-sub, html[data-text="large"] .topic-blurb,
html[data-text="large"] .activity li, html[data-text="large"] .facts, html[data-text="large"] .pr-conditions, html[data-text="large"] .sb-note { font-size: 15.5px; }
html[data-text="large"] .ob-task-title, html[data-text="large"] .rf-title, html[data-text="large"] .card-title, html[data-text="large"] .hs-title.q-btn { font-size: 18px; }
html[data-text="large"] .setting-label { font-size: 16px; }
html[data-text="large"] .pr-step-text { font-size: 20px; }
html[data-text="large"] .chip, html[data-text="large"] .kind, html[data-text="large"] .ob-status-pill, html[data-text="large"] .msg-meta { font-size: 13px; }
html[data-motion="reduced"] *, html[data-motion="reduced"] *::before, html[data-motion="reduced"] *::after {
    animation: none !important; transition: none !important; scroll-behavior: auto !important; }


/* ================================================================
   RESPONSIVE
   ================================================================ */
@media (max-width: 1100px) {
    .chat-inner { padding: 22px 20px 20px 20px; }
}
@media (max-width: 900px) {
    .sidebar { display: none; }
    .topbar-brand { display: grid; }
    .topbar-new-chat { display: flex; }
    .msg-row.user > .nicegui-html { max-width: 82%; }
    .landing-notes { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
    .topic-grid { grid-template-columns: 1fr; }
    .composer-foot { flex-wrap: wrap; }
    .answer-lang .lbl { display: none; }
    .topbar { padding: 0 12px; }
    .topbar-sub { display: none; }
    .status-pill .status-text { display: none; }
    .status-pill { padding: 0 10px; }
    .send-btn .q-btn__content .block { display: none; }
    .send-btn .q-btn__content .q-icon { margin: 0; }
    .send-btn.q-btn { width: 40px; padding: 0; border-radius: 50%; }
}
@media (max-width: 540px) {
    .sidebar { display: none; }
    .topbar-new-chat { display: flex; }
    .topbar { height: 56px; }
    .topbar-title { font-size: 15px; }
    .chat-inner { padding: 16px 12px 16px 12px; gap: 16px; }
    .msg-row { gap: 8px; }
    .msg-row.user > .nicegui-html { max-width: 90%; }
    .avatar { width: 30px; height: 30px; }
    .bubble { font-size: 14.5px; }
    .bubble.assistant { padding: 13px 14px; }
    .composer-wrap { padding: 8px 10px 10px 10px; }
    .composer-hint { display: none; }
    .hero h2 { font-size: 22px; }
    .dev-row { flex-direction: column; gap: 0; padding: 4px 0; }
    .dev-row .v { text-align: start; white-space: normal; word-break: break-word; max-width: 100%; }
}

/* PF-15 responsive rules: after the PF-10 blocks, so they refine them */
/* ---- below 900 px: the sidebar becomes a drawer opened by the menu button ---- */
@media (max-width: 900px) {
    .nav-toggle { display: inline-flex; }
    .drawer-close { display: inline-flex; position: absolute; top: 14px; inset-inline-end: 12px; }
    .app-shell.nav-open .sidebar { display: flex; position: fixed; top: 0; bottom: 0; inset-inline-start: 0; z-index: 70; width: min(320px, 88vw); min-width: 0;
        box-shadow: 0 18px 50px rgba(0, 0, 0, .3); animation: drawerIn 180ms ease both; }
    html[dir="rtl"] .app-shell.nav-open .sidebar { animation-name: drawerInRtl; }
    .app-shell.nav-open .nav-backdrop { display: block; position: fixed; inset: 0; z-index: 65; background: rgba(10, 20, 16, .45); animation: fadeIn 160ms ease both; }
    .sidebar .brand { padding-inline-end: 48px; }
    .page-main { padding: 22px 18px 48px 18px; }
    .grid-2 { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
    .composer-foot .answer-lang .lbl { display: inline; }
    .send-btn.q-btn .q-btn__content { font-size: 0; gap: 0; }
    .send-btn.q-btn .q-btn__content .q-icon { font-size: 20px; margin: 0; }
}
@media (max-width: 540px) {
    .topbar-history, .topbar-settings { display: none !important; }
    .topbar-left { gap: 8px; }
    .topbar .topbar-title { font-size: 14.5px; }
    .page-title { font-size: 22px; }
    .page-main { padding: 18px 14px 40px 14px; }
    .card { padding: 16px; }
    .setting-row { grid-template-columns: 1fr; }
    .seg.q-btn-group { width: 100%; }
    .seg.q-btn-group > .q-btn { flex: 1 1 0; padding: 4px 8px; }
    .hs-row { align-items: flex-start; }
    .facts { grid-template-columns: 1fr; gap: 0 0; }
    .facts dd { margin-bottom: 6px; }
}
@media (max-width: 400px) {
    .topbar-brand { display: none !important; }
    .topbar-right { gap: 6px; }
    .topbar-lang.q-btn { padding: 0 8px; min-width: 40px; }
}
</style>
"""


# Inline SVG icons (stroke = currentColor; sized by CSS)
_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"%s>%s</svg>'


def _icon(paths: str, cls: str = "") -> str:
    return _SVG % ((' class="%s"' % cls) if cls else "", paths)


ICONS = {
    "send": _icon('<path d="M12 19V5"/><path d="m5 12 7-7 7 7"/>'),
    "plus": _icon('<path d="M5 12h14"/><path d="M12 5v14"/>'),
    "sun": _icon('<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>', "sun"),
    "moon": _icon('<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>', "moon"),
    "copy": _icon('<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>'),
    "check": _icon('<path d="M20 6 9 17l-5-5"/>'),
    "chevron": _icon('<path d="m9 18 6-6-6-6"/>'),
    "info": _icon('<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>'),
    "alert": _icon('<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>'),
    "search": _icon('<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>'),
    "book": _icon('<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>'),
    "shield": _icon('<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>'),
    "clock": _icon('<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>'),
    "teaching": _icon('<path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/>'),
    "research": _icon('<path d="M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2"/><path d="M8.5 2h7"/><path d="M7 16h10"/>'),
    "services": _icon('<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>'),
    "institutional": _icon('<path d="M3 21h18"/><path d="M5 21V7l8-4v18"/><path d="M19 21V11l-6-4"/><path d="M9 9v.01"/><path d="M9 12v.01"/><path d="M9 15v.01"/><path d="M9 18v.01"/>'),
    "arrow": _icon('<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>'),
    "globe": _icon('<circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>'),
    # ---- PF-15 navigation and page icons (same 24 px stroke style) ----
    "chat": _icon('<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'),
    "list_checks": _icon('<path d="m3 17 2 2 4-4"/><path d="m3 7 2 2 4-4"/><path d="M13 6h8"/><path d="M13 12h8"/><path d="M13 18h8"/>'),
    "steps": _icon('<path d="M10 12h11"/><path d="M10 18h11"/><path d="M10 6h11"/><path d="M4 10h2"/><path d="M4 6h1v4"/><path d="M6 18H4c0-1 2-2 2-3s-1-1.5-2-1"/>'),
    "library": _icon('<path d="m16 6 4 14"/><path d="M12 6v14"/><path d="M8 8v12"/><path d="M4 4v16"/>'),
    "history": _icon('<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>'),
    "user": _icon('<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>'),
    "sliders": _icon('<line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/><line x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/><line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/><line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/><line x1="16" x2="16" y1="18" y2="22"/>'),
    "menu": _icon('<line x1="4" x2="20" y1="12" y2="12"/><line x1="4" x2="20" y1="6" y2="6"/><line x1="4" x2="20" y1="18" y2="18"/>'),
    "close": _icon('<path d="M18 6 6 18"/><path d="m6 6 12 12"/>'),
    "pencil": _icon('<path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/>'),
    "trash": _icon('<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>'),
    "external": _icon('<path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>'),
    "phone": _icon('<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>'),
    "file": _icon('<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/>'),
}
