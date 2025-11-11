(function () {
    // Opens nested details
    function openAncestors(id) {
        if (!id) return;
        const target = document.getElementById(decodeURIComponent(id));
        if (!target) return;
        let el = target;
        while (el && el !== document.body) {
            if (el.tagName && el.tagName.toLowerCase() === 'details') el.open = true;
            el = el.parentElement;
        }
        // Keep this instant to avoid fighting theme smooth scrolling
        target.scrollIntoView({ block: 'start', behavior: 'instant' });
    }

    function run() { openAncestors(location.hash.slice(1)); }

    document.addEventListener('DOMContentLoaded', run);
    window.addEventListener('hashchange', run);
})();