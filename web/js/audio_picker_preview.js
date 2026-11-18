import { app } from "../../../scripts/app.js";

// The frontend's upload-combo picker (FormDropdownMenuItem) renders audio-kind items as
// <img src="/api/view?filename=...">, which shows a broken image for every audio or video file.
// Swap those tiles for a play/pause button: audio files stream through one shared <audio>
// element, video files get their own <video> that shows a frame and plays in the tile.

const AUDIO_EXTENSIONS = new Set(["mp3", "wav", "flac", "ogg", "oga", "opus", "m4a", "aac", "weba"]);
const VIDEO_EXTENSIONS = new Set(["mp4", "m4v", "mov", "webm", "mkv"]);
const ITEM_IMAGE = '[data-testid="form-dropdown-menu"] img';
const TILE_CLASS = "yogurt-audio-preview";

const ICONS = {
    play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5.14v13.72a1 1 0 0 0 1.5.86l11-6.86a1 1 0 0 0 0-1.72l-11-6.86A1 1 0 0 0 8 5.14z"/></svg>',
    pause: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>',
    error: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>',
};

const audioPlayer = new Audio();
audioPlayer.preload = "none";
let activeTile = null;

app.registerExtension({
    name: "YogurtNodes.AudioPickerPreview",
    setup() {
        injectStyle();
        watchMedia(audioPlayer);

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
            if (activeTile && !activeTile.isConnected) stop();
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
    let tile = container.querySelector(`:scope > .${TILE_CLASS}`);
    const kind = mediaKind(img.src);

    // Vue reuses item elements when the list is filtered, so the same <img> can switch files.
    if (tile && (tile.dataset.kind !== kind || tile.dataset.src !== img.src)) {
        if (tile === activeTile) stop();
        tile.remove();
        tile = null;
    }
    if (!kind) {
        img.style.display = "";
        return;
    }
    img.style.display = "none";
    if (!tile) container.appendChild(createTile(img.src, kind));
}

function mediaKind(src) {
    let name = "";
    try {
        name = new URL(src, location.href).searchParams.get("filename") || "";
    } catch {
        return null;
    }
    const extension = name.slice(name.lastIndexOf(".") + 1).toLowerCase();
    if (!name.includes(".")) return null;
    if (VIDEO_EXTENSIONS.has(extension)) return "video";
    if (AUDIO_EXTENSIONS.has(extension)) return "audio";
    return null;
}

function createTile(src, kind) {
    const tile = document.createElement("div");
    tile.className = TILE_CLASS;
    tile.dataset.src = src;
    tile.dataset.kind = kind;
    if (kind === "video") {
        const video = document.createElement("video");
        video.preload = "metadata";
        video.playsInline = true;
        // A fragment past 0 makes the browser decode a real frame instead of a blank first one.
        video.src = `${src}#t=0.1`;
        watchMedia(video);
        tile.appendChild(video);
        tile.media = video;
    } else {
        tile.media = audioPlayer;
    }
    tile.insertAdjacentHTML("beforeend", `<button type="button"></button><div class="${TILE_CLASS}-progress"></div>`);

    const button = tile.querySelector("button");
    // Keep the click from reaching the dropdown item, which would select the file and close the menu.
    for (const type of ["pointerdown", "mousedown"]) {
        button.addEventListener(type, (event) => event.stopPropagation());
    }
    button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        toggle(tile);
    });
    setState(tile, "play");
    return tile;
}

function watchMedia(media) {
    const ifActive = (handler) => () => {
        if (activeTile?.media === media) handler(activeTile);
    };
    media.addEventListener("ended", ifActive((tile) => setState(tile, "play")));
    media.addEventListener("error", ifActive((tile) => setState(tile, "error")));
    media.addEventListener(
        "timeupdate",
        ifActive((tile) => media.duration && setProgress(tile, media.currentTime / media.duration))
    );
}

function toggle(tile) {
    const media = tile.media;
    if (tile === activeTile && !media.paused) {
        media.pause();
        setState(tile, "play");
        return;
    }
    if (tile !== activeTile) {
        stop();
        activeTile = tile;
        if (media === audioPlayer) audioPlayer.src = tile.dataset.src;
    }
    setState(tile, "pause");
    media.play().catch((error) => {
        if (error.name !== "AbortError") setState(tile, "error");
    });
}

function stop() {
    if (!activeTile) return;
    const media = activeTile.media;
    media.pause();
    if (media !== audioPlayer) media.currentTime = 0.1;
    setState(activeTile, "play");
    setProgress(activeTile, 0);
    activeTile = null;
}

function setState(tile, state) {
    const button = tile.querySelector("button");
    button.innerHTML = ICONS[state];
    button.title = state === "pause" ? "Pause" : state === "error" ? "Cannot play this file" : "Play";
    tile.dataset.state = state;
}

function setProgress(tile, fraction) {
    tile.querySelector(`.${TILE_CLASS}-progress`).style.width = `${fraction * 100}%`;
}

function injectStyle() {
    const style = document.createElement("style");
    style.textContent = `
.${TILE_CLASS} {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(to top right, #60a5fa, #14b8a6, #4ade80);
}
.${TILE_CLASS}[data-kind="video"] { background: #000; }
.${TILE_CLASS} > video { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
.${TILE_CLASS} > button {
    position: relative;
    width: 42%; height: 42%; min-width: 24px; min-height: 24px; max-width: 64px; max-height: 64px;
    display: flex; align-items: center; justify-content: center; padding: 0;
    border: none; border-radius: 9999px; cursor: pointer;
    background: rgba(0, 0, 0, 0.55); color: #fff; transition: background 0.15s, transform 0.15s, opacity 0.15s;
}
.${TILE_CLASS} > button:hover { background: rgba(0, 0, 0, 0.75); transform: scale(1.08); }
.${TILE_CLASS} > button svg { width: 50%; height: 50%; }
.${TILE_CLASS}[data-kind="video"][data-state="pause"] > button { opacity: 0; }
.${TILE_CLASS}[data-kind="video"][data-state="pause"]:hover > button { opacity: 1; }
.${TILE_CLASS}[data-state="error"] > button { background: rgba(185, 28, 28, 0.8); }
.${TILE_CLASS}-progress {
    position: absolute; left: 0; bottom: 0; height: 3px; width: 0; background: #fff; opacity: 0.9;
}`;
    document.head.appendChild(style);
}
