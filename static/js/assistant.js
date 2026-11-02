(() => {
    const launcher = document.getElementById("assistant-dock-launcher");
    const panel = document.getElementById("assistant-dock-panel");
    const closeButton = document.getElementById("assistant-dock-close");
    const frame = document.getElementById("assistant-dock-frame");

    if (!launcher || !panel || !closeButton || !frame) return;

    const dock = launcher.closest(".assistant-dock");
    const actionBar = document.querySelector(".action-bar");
    const DOCK_GAP = 12;

    function placeAboveActionBar() {
        dock.style.bottom = "";
        if (!actionBar || !panel.hidden) return;

        const bar = actionBar.getBoundingClientRect();
        const launcherBox = launcher.getBoundingClientRect();
        if (bar.top < launcherBox.bottom && bar.bottom > launcherBox.top) {
            const maxBottom = window.innerHeight - launcherBox.height - DOCK_GAP;
            const raised = Math.min(window.innerHeight - bar.top + DOCK_GAP, maxBottom);
            dock.style.bottom = `${raised}px`;
        }
    }

    function setOpen(open) {
        panel.hidden = !open;
        launcher.setAttribute("aria-expanded", String(open));
        if (open && !frame.src) frame.src = frame.dataset.src;
        if (!open) launcher.focus();
        placeAboveActionBar();
    }

    function closeOnEscape(event) {
        if (event.key === "Escape" && !panel.hidden) setOpen(false);
    }

    frame.addEventListener("load", () => {
        try {
            frame.contentWindow.document.addEventListener("keydown", closeOnEscape);
        } catch (error) {
            // Cross-origin chatbot: Escape only works from the parent page.
        }
    });

    window.addEventListener("scroll", placeAboveActionBar, { passive: true });
    window.addEventListener("resize", placeAboveActionBar);
    placeAboveActionBar();

    launcher.addEventListener("click", () => setOpen(panel.hidden));
    closeButton.addEventListener("click", () => setOpen(false));
    document.addEventListener("keydown", closeOnEscape);
})();