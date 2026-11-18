import { app } from "../../../scripts/app.js";

// The frontend's upload-combo picker (FormDropdownMenuItem) renders audio files as
// <img src="/api/view?filename=...">, which shows a broken image. Swap those tiles for a
// play/pause button that streams the same URL through one shared <audio> element.

const PLAYABLE_EXTENSIONS = new Set([
    "mp3", "wav", "flac", "ogg", "oga", "opus", "m4a", "aac", "weba", "webm", "mp4", "m4v", "mov",
]);
const ITEM_IMAGE = '[data-testid="form-dropdown-menu"] img';
const BUTTON_CLASS = "yogurt-audio-preview";

const ICONS = {
    play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5.14v13.72a1 1 0 0 0 1.5.86l11-6.86a1 1 0 0 0 0-1.72l-11-6.86A1 1 0 0 0 8 5.14z"/></svg>',
    pause: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>',
    error: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>',
};

const player = new Audio();
player.preload = "none";
let activeButton = null;

app.registerExtension({
    name: "YogurtNodes.AudioPickerPreview",
    setup() {
        injectStyle();
        player.addEventListener("ended", () => setState(activeButton, "play"));
        player.addEventListener("error", () => setState(activeButton, "error"));
        player.addEventListener("timeupdate", updateProgress);

        new MutationObserver((records) => {
            for (const record of records) {
                if (record.type === "attributes") {
                    if (record.target.matches(ITEM_IMAGE)) enhance(record.target);
                } else {
                    for (const node of record.addedNodes) {
                        if (node.nodeType === Node.ELEMENT_NODE) scan(node);
                    }
                }
            }
            if (activeButton && !activeButton.isConnected) stop();
        }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["src"] });
        scan(document.body);
    },
});

function scan(root) {
    if (root.matches(ITEM_IMAGE)) enhance(root);
    root.querySelectorAll(ITEM_IMAGE).forEach(enhance);
}

function enhance(img) {
    const container = img.parentElement;
    if (!container) return;
    let button = container.querySelector(`:scope > .${BUTTON_CLASS}`);

    // Vue reuses item elements when the list is filtered, so the same <img> can switch files.
    if (!isPlayable(img.src)) {
        img.style.display = "";
        if (button === activeButton) stop();
        button?.remove();
        return;
    }
    img.style.display = "none";
    if (!button) {
        button = createButton();
        container.appendChild(button);
    }
    if (button.dataset.src !== img.src) {
        if (button === activeButton) stop();
        button.dataset.src = img.src;
        setState(button, "play");
    }
}

function isPlayable(src) {
    let name = "";
    try {
        name = new URL(src, location.href).searchParams.get("filename") || "";
    } catch {
        return false;
    }
    const dot = name.lastIndexOf(".");
    return dot !== -1 && PLAYABLE_EXTENSIONS.has(name.slice(dot + 1).toLowerCase());
}

function createButton() {
    const button = document.createElement("div");
    button.className = BUTTON_CLASS;
    button.innerHTML = `<button type="button"></button><div class="${BUTTON_CLASS}-progress"></div>`;
    const toggleButton = button.querySelector("button");
    // Keep the click from reaching the dropdown item, which would select the file and close the menu.
    for (const type of ["pointerdown", "mousedown"]) {
        toggleButton.addEventListener(type, (event) => event.stopPropagation());
    }
    toggleButton.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        toggle(button);
    });
    return button;
}

function toggle(button) {
    if (button === activeButton && !player.paused) {
        player.pause();
        setState(button, "play");
        return;
    }
    if (button !== activeButton) {
        stop();
        activeButton = button;
        player.src = button.dataset.src;
    }
    setState(button, "pause");
    player.play().catch((error) => {
        if (error.name !== "AbortError") setState(button, "error");
    });
}

function stop() {
    player.pause();
    if (activeButton) {
        setState(activeButton, "play");
        setProgress(activeButton, 0);
    }
    activeButton = null;
}

function setState(button, state) {
    if (!button) return;
    const toggleButton = button.querySelector("button");
    toggleButton.innerHTML = ICONS[state];
    toggleButton.title = state === "pause" ? "Pause" : state === "error" ? "Cannot play this file" : "Play";
    button.dataset.state = state;
}

function updateProgress() {
    if (activeButton && player.duration) setProgress(activeButton, player.currentTime / player.duration);
}

function setProgress(button, fraction) {
    button.querySelector(`.${BUTTON_CLASS}-progress`).style.width = `${fraction * 100}%`;
}

function injectStyle() {
    const style = document.createElement("style");
    style.textContent = `
.${BUTTON_CLASS} {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(to top right, #60a5fa, #14b8a6, #4ade80);
}
.${BUTTON_CLASS} > button {
    width: 42%; height: 42%; min-width: 24px; min-height: 24px; max-width: 64px; max-height: 64px;
    display: flex; align-items: center; justify-content: center; padding: 0;
    border: none; border-radius: 9999px; cursor: pointer;
    background: rgba(0, 0, 0, 0.55); color: #fff; transition: background 0.15s, transform 0.15s;
}
.${BUTTON_CLASS} > button:hover { background: rgba(0, 0, 0, 0.75); transform: scale(1.08); }
.${BUTTON_CLASS} > button svg { width: 50%; height: 50%; }
.${BUTTON_CLASS}[data-state="error"] > button { background: rgba(185, 28, 28, 0.8); }
.${BUTTON_CLASS}-progress {
    position: absolute; left: 0; bottom: 0; height: 3px; width: 0; background: #fff; opacity: 0.9;
}`;
    document.head.appendChild(style);
}
