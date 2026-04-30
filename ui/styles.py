"""
Theme + CSS for the UOS Faculty Onboarding Chatbot UI.

Design language:
    - Official UOS brand palette: forest green, deep black, clean white.
    - Controls are pulled up to the topbar (theme + dev mode live top-right).
    - Wider, adaptive chat surface so bubbles breathe.
    - Tailored light AND dark palettes, both built around the UOS green.
"""

GLOBAL_CSS = """
<style>
/* ================================================================
   THEME TOKENS  -  UOS GREEN / BLACK / WHITE
   ================================================================ */
:root {
    /* UOS brand palette (sampled from the official logo) */
    --uos-green:        #0B8A5A;   /* primary brand green */
    --uos-green-deep:   #06633F;   /* darker shade for hover / pressed */
    --uos-green-bright: #1FB37A;   /* lighter shade for dark-mode accents */
    --uos-green-soft:   rgba(11, 138, 90, 0.14);
    --uos-green-ring:   rgba(11, 138, 90, 0.32);

    --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto,
            Oxygen, Ubuntu, sans-serif;

    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;

    --transition: 180ms cubic-bezier(0.2, 0.8, 0.2, 1);

    /* Shared accent alias - swapped per theme below */
    --uos: var(--uos-green);
    --uos-soft: var(--uos-green-soft);
    --uos-ring: var(--uos-green-ring);
}

/* ================================================================
   DARK THEME  -  deep black canvas, green accents, white text
   ================================================================ */
:root,
[data-theme="dark"] {
    --bg-0: #0a0c0b;   /* near-black with a hint of green undertone */
    --bg-1: #111413;
    --bg-2: #171b19;
    --bg-3: #1f2422;
    --bg-hover: #252b28;

    --border: rgba(255, 255, 255, 0.07);
    --border-strong: rgba(255, 255, 255, 0.14);
    --border-focus: rgba(31, 179, 122, 0.45);

    --text-0: #ffffff;
    --text-1: rgba(255, 255, 255, 0.76);
    --text-2: rgba(255, 255, 255, 0.52);
    --text-3: rgba(255, 255, 255, 0.32);

    /* User bubble: solid UOS green on dark */
    --user-bubble: var(--uos-green);
    --user-bubble-text: #ffffff;

    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 4px 14px rgba(0, 0, 0, 0.42);
    --shadow-lg: 0 12px 32px rgba(0, 0, 0, 0.50);

    --logo-bg: #ffffff;
    --logo-ring: rgba(255, 255, 255, 0.10);

    --ok: var(--uos-green-bright);
    --warn: #fbbf24;
    --danger: #f87171;

    /* Use brighter green for dark-mode accents so it pops on black */
    --uos: var(--uos-green-bright);
    --uos-soft: rgba(31, 179, 122, 0.16);
    --uos-ring: rgba(31, 179, 122, 0.36);
}

/* ================================================================
   LIGHT THEME  -  clean white, green accents, deep black text
   ================================================================ */
[data-theme="light"] {
    --bg-0: #fafbfa;    /* paper-white with a subtle green warmth */
    --bg-1: #ffffff;
    --bg-2: #f2f5f3;
    --bg-3: #e8ede9;
    --bg-hover: #e0e7e2;

    --border: rgba(6, 99, 63, 0.08);
    --border-strong: rgba(6, 99, 63, 0.18);
    --border-focus: rgba(11, 138, 90, 0.40);

    --text-0: #0a1410;
    --text-1: rgba(10, 20, 16, 0.76);
    --text-2: rgba(10, 20, 16, 0.54);
    --text-3: rgba(10, 20, 16, 0.34);

    /* User bubble: solid UOS green on light (white text) */
    --user-bubble: var(--uos-green);
    --user-bubble-text: #ffffff;

    --shadow-sm: 0 1px 2px rgba(6, 99, 63, 0.06);
    --shadow-md: 0 4px 14px rgba(6, 99, 63, 0.10);
    --shadow-lg: 0 12px 32px rgba(6, 99, 63, 0.12);

    --logo-bg: transparent;
    --logo-ring: rgba(6, 99, 63, 0.08);

    --ok: var(--uos-green);
    --warn: #d97706;
    --danger: #dc2626;

    --uos: var(--uos-green);
    --uos-soft: var(--uos-green-soft);
    --uos-ring: var(--uos-green-ring);
}

/* ================================================================
   RESET / BASE
   ================================================================ */
html, body, #q-app {
    background: var(--bg-0) !important;
    color: var(--text-0) !important;
    font-family: var(--sans);
    font-feature-settings: "ss01" on, "cv11" on;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    transition: background var(--transition), color var(--transition);
}

* { box-sizing: border-box; }

/* Remove NiceGUI's default page padding */
.q-page-container, .q-page { padding: 0 !important; }
header.q-header, footer.q-footer { display: none !important; }
.nicegui-content { padding: 0 !important; gap: 0 !important; max-width: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--border-strong);
    border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover { background: var(--uos-ring); }

/* Text selection uses brand green */
::selection { background: var(--uos-soft); color: var(--text-0); }

/* ================================================================
   QUASAR COLOR OVERRIDES  -  neutralise default primary-blue bleed
   ================================================================ */
/* NiceGUI/Quasar applies a default "primary" colour (blue) to q-btn
   text and any .q-icon/svg inside. Strip it globally for our shell so
   our green / monochrome buttons don't come out looking blue. */
.q-btn .q-btn__content { color: inherit !important; }
.q-btn .q-focus-helper { display: none !important; }
.q-ripple { display: none !important; }

/* ================================================================
   LAYOUT SHELL
   ================================================================ */
.app-shell {
    display: flex;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
    background: var(--bg-0);
}

/* ================================================================
   SIDEBAR
   ================================================================ */
.sidebar {
    width: 280px;
    min-width: 280px;
    background: var(--bg-1);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 18px 14px 14px 14px;
    gap: 14px;
    overflow-y: auto;
    transition: background var(--transition), border-color var(--transition);
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 6px 10px 6px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 4px;
}
.brand-logo {
    width: 44px;
    height: 44px;
    border-radius: 10px;
    background: var(--logo-bg);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 0 1px var(--logo-ring);
    flex-shrink: 0;
    padding: 4px;
    transition: background var(--transition), box-shadow var(--transition);
}
.brand-logo img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
}
.brand-text {
    display: flex;
    flex-direction: column;
    min-width: 0;
}
.brand-title {
    font-size: 13.5px;
    font-weight: 650;
    letter-spacing: -0.01em;
    color: var(--text-0);
    line-height: 1.2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.brand-subtitle {
    font-size: 10.5px;
    color: var(--uos);
    margin-top: 3px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}

/* "New chat" primary button  -  UOS green */
.new-chat-btn {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 11px 14px;
    border: 1px solid var(--uos);
    border-radius: var(--radius-md);
    background: var(--uos) !important;
    color: #ffffff !important;
    font-size: 13.5px;
    font-weight: 600;
    cursor: pointer;
    transition: all var(--transition);
    font-family: inherit;
    box-shadow: 0 1px 3px rgba(11, 138, 90, 0.18);
}
/* Quasar wraps button children in .q-btn__content - make sure that wrapper
   centers our icon + label as a single flex group, not spread apart. */
.new-chat-btn .q-btn__content {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    width: 100%;
    flex-wrap: nowrap !important;
    color: #ffffff !important;
}
.new-chat-btn .q-btn__content *,
.new-chat-btn span,
.new-chat-btn svg,
.new-chat-btn svg * {
    color: #ffffff !important;
    stroke: #ffffff !important;
}
.new-chat-btn:hover {
    background: var(--uos-green-deep) !important;
    border-color: var(--uos-green-deep);
    transform: translateY(-0.5px);
    box-shadow: 0 4px 12px rgba(11, 138, 90, 0.28);
}
[data-theme="dark"] .new-chat-btn:hover {
    background: var(--uos-green) !important;
    border-color: var(--uos-green);
}
.new-chat-btn:active { transform: translateY(0); }
.new-chat-btn .plus {
    width: 16px; height: 16px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    font-size: 13px;
    line-height: 1;
    color: #ffffff !important;
    flex-shrink: 0;
}
.new-chat-btn .plus svg {
    width: 14px;
    height: 14px;
    stroke: #ffffff !important;
    stroke-width: 2.4;
}

/* Section label */
.sb-label {
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-3);
    padding: 8px 6px 4px 6px;
}

/* Session / history item */
.session-item {
    padding: 9px 12px;
    border-radius: var(--radius-sm);
    color: var(--text-1);
    font-size: 13px;
    cursor: pointer;
    transition: all 140ms ease;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    border: 1px solid transparent;
    border-left: 2px solid transparent;
}
.session-item:hover {
    background: var(--bg-2);
    color: var(--text-0);
    border-left-color: var(--uos-ring);
}
.session-item.active {
    background: var(--uos-soft);
    color: var(--text-0);
    border-color: var(--uos-ring);
    border-left: 2px solid var(--uos);
    font-weight: 500;
}

/* Sidebar footer block */
.sb-footer {
    margin-top: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
}

/* Status pills */
.status-block {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 12px 12px;
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    font-size: 11.5px;
    color: var(--text-2);
}
.status-line {
    display: flex;
    align-items: center;
    gap: 8px;
}
.status-line .label { color: var(--text-3); min-width: 50px; font-weight: 500; }
.status-line .value {
    color: var(--text-1);
    font-family: var(--mono);
    font-size: 11px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--text-3);
    flex-shrink: 0;
}
.status-dot.ok {
    background: var(--uos);
    box-shadow: 0 0 0 3px var(--uos-soft);
}
.status-dot.warn {
    background: var(--warn);
    box-shadow: 0 0 0 3px rgba(251, 191, 36, 0.16);
}

/* ================================================================
   MAIN AREA
   ================================================================ */
.main {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--bg-0);
    position: relative;
    min-width: 0;
}

/* Topbar now hosts title on the LEFT and controls on the RIGHT:
   theme toggle, developer-mode toggle, and status pill. */
.topbar {
    height: 58px;
    padding: 0 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    background: var(--bg-0);
    flex-shrink: 0;
    gap: 14px;
    transition: border-color var(--transition);
}
.topbar-left {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
}
.topbar-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-0);
    letter-spacing: -0.01em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.topbar-accent {
    width: 3px;
    height: 18px;
    background: var(--uos);
    border-radius: 2px;
    flex-shrink: 0;
}

.topbar-right {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
}

.topbar-pill {
    font-size: 11px;
    padding: 5px 11px;
    border-radius: 999px;
    background: var(--uos-soft);
    border: 1px solid var(--uos-ring);
    color: var(--uos);
    font-family: var(--mono);
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.topbar-pill .status-dot.ok {
    background: var(--uos);
    box-shadow: 0 0 0 3px var(--uos-soft);
}

/* Control button in topbar (theme toggle, etc.) */
.topbar-btn {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    border: 1px solid var(--border-strong);
    background: var(--bg-1) !important;
    color: var(--text-1) !important;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all var(--transition);
    flex-shrink: 0;
    padding: 0;
}
.topbar-btn .q-btn__content,
.topbar-btn .q-btn__content *,
.topbar-btn svg,
.topbar-btn svg * {
    color: inherit !important;
    stroke: currentColor !important;
}
.topbar-btn:hover {
    background: var(--bg-2) !important;
    color: var(--uos) !important;
    border-color: var(--uos-ring);
    transform: translateY(-0.5px);
}
.topbar-btn:active { transform: translateY(0); }
.topbar-btn svg {
    width: 17px;
    height: 17px;
    stroke-width: 1.9;
}
.topbar-btn .sun  { display: none; }
.topbar-btn .moon { display: block; }
[data-theme="light"] .topbar-btn .sun  { display: block; }
[data-theme="light"] .topbar-btn .moon { display: none; }

/* Dev-mode toggle as a compact labeled pill in the topbar */
.dev-toggle-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px 6px 14px;
    height: 36px;
    border-radius: 999px;
    border: 1px solid var(--border-strong);
    background: var(--bg-1);
    color: var(--text-1);
    font-size: 12px;
    font-weight: 500;
    transition: all var(--transition);
    user-select: none;
}
.dev-toggle-pill:hover {
    border-color: var(--uos-ring);
}
.dev-toggle-pill.active {
    background: var(--uos-soft);
    border-color: var(--uos-ring);
    color: var(--uos);
}
.dev-toggle-pill .dev-label {
    letter-spacing: 0.01em;
    white-space: nowrap;
}
.dev-toggle-pill .q-toggle {
    padding: 0 !important;
    min-height: 0 !important;
}

/* Chat scroll + adaptive width */
.chat-scroll {
    flex: 1;
    overflow-y: auto;
    scroll-behavior: smooth;
}
.chat-inner {
    max-width: 1100px;        /* much wider than before */
    width: 100%;
    margin: 0 auto;
    padding: 32px 40px 180px 40px;
}

/* ================================================================
   MESSAGE BUBBLES  -  wider, more breathable
   ================================================================ */
.msg-row {
    display: flex;
    margin: 18px 0;
    gap: 14px;
    align-items: flex-start;
    animation: fadeUp 0.24s cubic-bezier(0.2, 0.8, 0.2, 1);
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
.msg-row.user      {
    justify-content: flex-end;
    flex-wrap: nowrap;
}
.msg-row.assistant { justify-content: flex-start; }

/* NiceGUI wraps every ui.html(...) call in its own .nicegui-html div.
   Inside .msg-row (which is flex), those wrappers become the flex
   children - NOT our .bubble / .avatar. So the wrapper is what we must
   size correctly. We cap the bubble's wrapper at 72% and let it shrink
   to fit its content; the avatar's wrapper stays at its natural width. */
.msg-row > * {
    min-width: 0;
}
/* User row: [bubble-wrapper, avatar-wrapper] */
.msg-row.user > *:first-child {
    flex: 0 1 auto;
    max-width: 72%;
}
.msg-row.user > *:last-child {
    flex: 0 0 auto;
}
/* Assistant row: [avatar-wrapper, assistant-wrap] */
.msg-row.assistant > *:first-child {
    flex: 0 0 auto;
}
.msg-row.assistant > *:nth-child(2) {
    flex: 1 1 auto;
    min-width: 0;
}

.avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 4px;
    letter-spacing: 0.03em;
}
.avatar.assistant {
    background: #ffffff;
    color: var(--uos);
    border: 1px solid var(--uos-ring);
    padding: 4px;
    box-shadow: 0 2px 6px rgba(11, 138, 90, 0.12);
}
.avatar.assistant img {
    width: 100%; height: 100%;
    object-fit: contain;
    display: block;
}
.avatar.user {
    background: var(--uos);
    color: #ffffff;
    border: 1px solid var(--uos-green-deep);
    font-size: 10.5px;
}
[data-theme="dark"] .avatar.user {
    border-color: var(--uos-green);
}

.bubble {
    padding: 13px 18px;
    border-radius: var(--radius-lg);
    font-size: 14.5px;
    line-height: 1.65;
    word-wrap: break-word;
    overflow-wrap: break-word;
}
/* User bubble: solid UOS green. Using inline-block so the bubble sizes
   to its text content naturally and wraps between words - works the
   same regardless of how deeply the NiceGUI wrappers nest it. */
.bubble.user {
    background: var(--user-bubble);
    color: var(--user-bubble-text);
    border-top-right-radius: 6px;
    display: inline-block;
    max-width: 100%;
    white-space: normal;
    word-break: normal;
    overflow-wrap: break-word;
    text-align: left;
    box-shadow: 0 2px 8px rgba(11, 138, 90, 0.18);
}
.bubble.assistant {
    background: var(--bg-1);
    color: var(--text-0);
    border: 1px solid var(--border);
    border-top-left-radius: 6px;
    box-shadow: var(--shadow-sm);
    width: fit-content;
    max-width: 100%;
}

.assistant-wrap {
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex: 1;
    min-width: 0;
    max-width: calc(100% - 46px);
}

/* Markdown inside assistant bubbles */
.bubble.assistant .nicegui-markdown > *:first-child { margin-top: 0; }
.bubble.assistant .nicegui-markdown > *:last-child  { margin-bottom: 0; }
.bubble.assistant .nicegui-markdown p { margin: 0.5em 0; }
.bubble.assistant .nicegui-markdown code {
    background: var(--bg-2);
    border: 1px solid var(--border);
    padding: 1px 6px;
    border-radius: 5px;
    font-family: var(--mono);
    font-size: 0.88em;
    color: var(--uos);
}
.bubble.assistant .nicegui-markdown pre {
    background: var(--bg-2);
    border: 1px solid var(--border);
    padding: 12px 14px;
    border-radius: var(--radius-md);
    overflow-x: auto;
    font-family: var(--mono);
    font-size: 0.88em;
    line-height: 1.55;
}
.bubble.assistant .nicegui-markdown pre code {
    background: transparent;
    border: none;
    padding: 0;
    color: inherit;
}
.bubble.assistant .nicegui-markdown ul,
.bubble.assistant .nicegui-markdown ol {
    padding-left: 22px;
    margin: 0.4em 0;
}
.bubble.assistant .nicegui-markdown blockquote {
    border-left: 3px solid var(--uos);
    padding: 8px 12px;
    color: var(--text-1);
    margin: 0.6em 0;
    background: var(--uos-soft);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
}
.bubble.assistant .nicegui-markdown a {
    color: var(--uos);
    text-decoration: none;
    border-bottom: 1px dotted var(--uos-ring);
    font-weight: 500;
}
.bubble.assistant .nicegui-markdown a:hover { border-bottom-style: solid; }
.bubble.assistant .nicegui-markdown strong { color: var(--text-0); font-weight: 600; }

/* ================================================================
   SOURCE CHIPS + TOOLBAR
   ================================================================ */
.source-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    margin-top: 2px;
}
.chip {
    font-size: 11.5px;
    padding: 4px 10px;
    border-radius: 999px;
    background: var(--bg-2);
    color: var(--text-1);
    border: 1px solid var(--border);
    font-variant-numeric: tabular-nums;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
    font-weight: 500;
}
.chip.page {
    background: var(--uos-soft);
    color: var(--uos);
    border-color: var(--uos-ring);
    font-weight: 600;
}
.chip.section {
    max-width: 360px;
    overflow: hidden;
    text-overflow: ellipsis;
}
.chip.ocr {
    background: var(--bg-3);
    color: var(--text-1);
    font-size: 10.5px;
}

.msg-toolbar {
    display: flex;
    gap: 2px;
    padding-left: 2px;
    opacity: 0.65;
    transition: opacity 140ms ease;
}
.assistant-wrap:hover .msg-toolbar { opacity: 1; }
.icon-btn {
    background: transparent;
    border: 1px solid transparent;
    color: var(--text-2);
    padding: 5px 11px;
    font-size: 11.5px;
    border-radius: 6px;
    cursor: pointer;
    transition: all var(--transition);
    font-family: inherit;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.icon-btn:hover {
    background: var(--uos-soft);
    color: var(--uos);
    border-color: var(--uos-ring);
}
.icon-btn svg { width: 13px; height: 13px; stroke-width: 1.9; }

/* ================================================================
   EMPTY STATE / LANDING
   ================================================================ */
.landing {
    max-width: 760px;
    margin: 72px auto 0 auto;
    text-align: center;
    padding: 0 16px;
    animation: fadeUp 0.35s cubic-bezier(0.2, 0.8, 0.2, 1);
}
.landing-logo {
    width: 72px; height: 72px;
    margin: 0 auto 22px auto;
    background: var(--logo-bg);
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 10px;
    box-shadow: var(--shadow-md), 0 0 0 1px var(--logo-ring);
}
.landing-logo img {
    width: 100%; height: 100%;
    object-fit: contain;
    display: block;
}
.landing-title {
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: var(--text-0);
    margin-bottom: 10px;
}
.landing-title .accent { color: var(--uos); }
.landing-subtitle {
    color: var(--text-2);
    font-size: 14.5px;
    line-height: 1.6;
    margin-bottom: 40px;
    max-width: 560px;
    margin-left: auto;
    margin-right: auto;
}
.example-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    text-align: left;
}
.example-card {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 16px 18px;
    color: var(--text-1);
    font-size: 13.5px;
    line-height: 1.5;
    cursor: pointer;
    transition: all var(--transition);
    position: relative;
    border-left: 3px solid transparent;
}
.example-card:hover {
    background: var(--bg-2);
    color: var(--text-0);
    border-color: var(--border-strong);
    border-left-color: var(--uos);
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}
.example-card .example-label {
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--uos);
    margin-bottom: 6px;
    font-weight: 700;
}

/* ================================================================
   TYPING INDICATOR
   ================================================================ */
.typing {
    display: inline-flex; gap: 4px; padding: 6px 2px; align-items: center;
}
.typing span {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--uos);
    animation: typingPulse 1.2s infinite ease-in-out;
}
.typing span:nth-child(2) { animation-delay: 0.15s; }
.typing span:nth-child(3) { animation-delay: 0.30s; }
@keyframes typingPulse {
    0%, 80%, 100% { opacity: 0.25; transform: scale(0.75); }
    40% { opacity: 1; transform: scale(1); }
}

/* ================================================================
   COMPOSER (sticky bottom)  -  wider, green focus ring
   ================================================================ */
.composer-wrap {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    padding: 16px 32px 24px 32px;
    background: linear-gradient(180deg,
        rgba(0,0,0,0) 0%,
        var(--bg-0) 40%);
    pointer-events: none;
    transition: background var(--transition);
}
.composer {
    max-width: 1040px;        /* widened to match chat area */
    margin: 0 auto;
    display: flex;
    gap: 10px;
    align-items: flex-end;
    background: var(--bg-1);
    border: 1.5px solid var(--border-strong);
    border-radius: var(--radius-lg);
    padding: 10px 10px 10px 18px;
    box-shadow: var(--shadow-md);
    pointer-events: auto;
    transition: border-color var(--transition), box-shadow var(--transition);
}
.composer:focus-within {
    border-color: var(--uos);
    box-shadow: var(--shadow-md), 0 0 0 3px var(--uos-soft);
}
.composer textarea {
    flex: 1;
    background: transparent !important;
    color: var(--text-0) !important;
    border: none !important;
    outline: none !important;
    resize: none;
    font-size: 14.5px;
    line-height: 1.55;
    max-height: 200px;
    padding: 8px 0 !important;
    font-family: inherit;
}
.composer textarea::placeholder { color: var(--text-3); }

.send-btn {
    background: var(--uos) !important;
    color: #ffffff !important;
    border: none;
    border-radius: 10px;
    width: 38px;
    height: 38px;
    display: flex !important;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all var(--transition);
    flex-shrink: 0;
    padding: 0;
    box-shadow: 0 2px 6px rgba(11, 138, 90, 0.22);
}
/* Kill Quasar default blue on the inner SVG arrow */
.send-btn .q-btn__content,
.send-btn .q-btn__content *,
.send-btn svg,
.send-btn svg * {
    color: #ffffff !important;
    stroke: #ffffff !important;
}
.send-btn:hover:not(:disabled) {
    background: var(--uos-green-deep) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(11, 138, 90, 0.32);
}
[data-theme="dark"] .send-btn:hover:not(:disabled) {
    background: var(--uos-green) !important;
}
.send-btn:active:not(:disabled) { transform: translateY(0); }
.send-btn:disabled {
    background: var(--bg-3) !important;
    color: var(--text-3) !important;
    cursor: not-allowed;
    box-shadow: none;
}
.send-btn:disabled svg,
.send-btn:disabled svg * {
    stroke: var(--text-3) !important;
}
.send-btn svg {
    width: 16px;
    height: 16px;
    stroke-width: 2.2;
}

.composer-hint {
    max-width: 1040px;
    margin: 10px auto 0 auto;
    font-size: 11px;
    color: var(--text-3);
    text-align: center;
    font-family: var(--mono);
    letter-spacing: 0.02em;
    pointer-events: auto;
}
.composer-hint kbd {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-bottom-width: 2px;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 10px;
    font-family: var(--mono);
    color: var(--text-1);
}

/* ================================================================
   DEVELOPER-MODE PANEL
   ================================================================ */
.dev-panel {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-left: 3px solid var(--uos);
    border-radius: var(--radius-md);
    overflow: hidden;
    font-family: var(--mono);
    font-size: 11.5px;
}
.dev-panel details { padding: 0; }
.dev-panel summary {
    padding: 10px 14px;
    cursor: pointer;
    color: var(--uos);
    list-style: none;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
    user-select: none;
    transition: background var(--transition);
}
.dev-panel summary::-webkit-details-marker { display: none; }
.dev-panel summary:hover { background: var(--uos-soft); }
.dev-panel summary::before {
    content: "›";
    display: inline-block;
    transition: transform var(--transition);
    font-size: 14px;
    color: var(--uos);
}
.dev-panel details[open] summary::before { transform: rotate(90deg); }
.dev-panel details[open] summary { border-bottom: 1px solid var(--border); }

.dev-body { padding: 10px 14px 12px 14px; }
.dev-section {
    font-size: 9.5px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--uos);
    margin: 10px 0 4px 0;
    font-weight: 700;
}
.dev-section:first-child { margin-top: 0; }
.dev-row {
    display: flex;
    justify-content: space-between;
    padding: 2px 0;
    gap: 10px;
}
.dev-row .k { color: var(--text-2); }
.dev-row .v {
    color: var(--text-0);
    font-variant-numeric: tabular-nums;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ================================================================
   MINI UI PRIMITIVES
   ================================================================ */
/* NiceGUI / Quasar toggle styling to use brand green */
.q-toggle__inner--falsy { color: var(--text-2) !important; }
.q-toggle__inner--truthy { color: var(--uos) !important; }
.q-toggle__thumb { box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important; }

/* Banner */
.banner {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-left: 3px solid var(--warn);
    border-radius: var(--radius-md);
    padding: 12px 14px;
    margin-bottom: 20px;
    font-size: 13px;
    color: var(--text-1);
    line-height: 1.5;
}
.banner.danger { border-left-color: var(--danger); }
.banner.info   { border-left-color: var(--uos); background: var(--uos-soft); }
.banner strong { color: var(--text-0); }

/* ================================================================
   RESPONSIVE  -  chat adapts down to mobile
   ================================================================ */
@media (max-width: 1180px) {
    .chat-inner  { max-width: 900px; padding: 28px 32px 170px 32px; }
    .composer    { max-width: 860px; }
    .composer-hint { max-width: 860px; }
}

@media (max-width: 960px) {
    .chat-inner  { max-width: 100%; padding: 24px 24px 160px 24px; }
    .composer    { max-width: 100%; }
    .composer-hint { max-width: 100%; }
    .composer-wrap { padding: 14px 22px 20px 22px; }
}

@media (max-width: 820px) {
    .sidebar { width: 236px; min-width: 236px; padding: 14px 10px; }
    .brand-title { font-size: 12.5px; }
    .chat-inner { padding: 20px 16px 170px 16px; }
    .landing-title { font-size: 24px; }
    .example-grid { grid-template-columns: 1fr; }
    .bubble { font-size: 14px; }
    .bubble.user { max-width: 86%; }
    .composer-wrap { padding: 12px 14px 18px 14px; }
    .topbar { padding: 0 14px; height: 52px; }
    .topbar-title { font-size: 13px; }
    /* Hide model name on small screens to save space */
    .topbar-pill { display: none; }
    .dev-toggle-pill .dev-label { display: none; }
    .dev-toggle-pill { padding: 6px 8px; }
}

@media (max-width: 540px) {
    .sidebar { display: none; }
}
</style>
"""


# SVG icons (inline, theme-aware via currentColor)
ICONS = {
    "send": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<path d="M12 19V5"/>'
        '<path d="m5 12 7-7 7 7"/>'
        '</svg>'
    ),
    "plus": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<path d="M5 12h14"/><path d="M12 5v14"/>'
        '</svg>'
    ),
    "sun": (
        '<svg class="sun" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="4"/>'
        '<path d="M12 2v2"/><path d="M12 20v2"/>'
        '<path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/>'
        '<path d="M2 12h2"/><path d="M20 12h2"/>'
        '<path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>'
        '</svg>'
    ),
    "moon": (
        '<svg class="moon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>'
        '</svg>'
    ),
    "copy": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/>'
        '<path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>'
        '</svg>'
    ),
    "code": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<polyline points="16 18 22 12 16 6"/>'
        '<polyline points="8 6 2 12 8 18"/>'
        '</svg>'
    ),
}
