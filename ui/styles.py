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
}
