// panelFade.js
// Adds fade-in/fade-out behavior for page cards and exposes navigateWithFade(url).
(function () {
    // Insert CSS rules for fade
    const css = `
    .card { opacity: 0; transform: translateY(8px); transition: opacity 280ms ease, transform 280ms ease; }
    .card.fade-in { opacity: 1; transform: translateY(0); }
    .card.fade-out { opacity: 0; transform: translateY(8px); }
    `;
    const style = document.createElement('style');
    style.setAttribute('data-generated-by', 'panelFade');
    style.appendChild(document.createTextNode(css));
    document.head.appendChild(style);

    // Fade in on load
    function fadeInCard() {
        const card = document.querySelector('.card');
        if (!card) return;
        // Ensure we trigger transition
        requestAnimationFrame(() => {
            card.classList.remove('fade-out');
            card.classList.add('fade-in');
        });
    }

    // Navigate after fade-out
    function navigateWithFade(url) {
        const card = document.querySelector('.card');
        if (!card) {
            window.location.href = url;
            return;
        }
        // If already fading out, ignore
        if (card.classList.contains('fade-out')) return;
        // Start fade-out
        card.classList.remove('fade-in');
        card.classList.add('fade-out');

        // Wait for transition end (fallback timeout 350ms)
        let done = false;
        function go() {
            if (done) return; done = true;
            window.location.href = url;
        }
        card.addEventListener('transitionend', (ev) => {
            if (ev.propertyName === 'opacity') go();
        }, { once: true });
        setTimeout(go, 400);
    }

    // Expose globally
    window.navigateWithFade = navigateWithFade;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', fadeInCard);
    } else {
        fadeInCard();
    }
})();
