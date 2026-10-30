(() => {
    const launcher = document.getElementById("assistant-dock-launcher");
    const panel = document.getElementById("assistant-dock-panel");
    const closeButton = document.getElementById("assistant-dock-close");
    const frame = document.getElementById("assistant-dock-frame");

    if (!launcher || !panel || !closeButton || !frame) return;

    function setOpen(open) {
        panel.hidden = !open;
        launcher.setAttribute("aria-expanded", String(open));
        if (open && !frame.src) frame.src = frame.dataset.src;
        if (!open) launcher.focus();
    }

    launcher.addEventListener("click", () => setOpen(panel.hidden));
    closeButton.addEventListener("click", () => setOpen(false));
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && !panel.hidden) setOpen(false);
    });
})();