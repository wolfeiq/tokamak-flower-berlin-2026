'use strict';
/* Presentation shell. Slides are every <section class="slide"> in DOM order —
 * add one and it gains navigation, counter and progress with no registration.
 * Arrow keys advance except while typing in a form control, so the interactive
 * elements on each slide stay usable mid-talk. */
(() => {
  let current = 0, slides = [];
  const pad = (n) => String(n).padStart(2, '0');
  function show(index) {
    current = Math.max(0, Math.min(index, slides.length - 1));
    slides.forEach((slide, i) => slide.classList.toggle('active', i === current));
    const progress = document.getElementById('deck-progress');
    if (progress) progress.style.width = ((current + 1) / slides.length * 100) + '%';
    const counter = document.getElementById('deck-counter');
    if (counter) counter.textContent = `${pad(current + 1)} / ${pad(slides.length)}`;
    if (slides[current].id) history.replaceState(null, '', '#' + slides[current].id);
  }
  function typing(target) {
    return target && /^(input|textarea|select|button)$/i.test(target.tagName);
  }
  function init() {
    slides = [...document.querySelectorAll('section.slide')];
    if (!slides.length) return;
    const fromHash = slides.findIndex(s => '#' + s.id === location.hash);
    document.getElementById('deck-next')?.addEventListener('click', () => show(current + 1));
    document.getElementById('deck-prev')?.addEventListener('click', () => show(current - 1));
    document.addEventListener('keydown', (event) => {
      if (typing(event.target)) return;
      if (event.key === 'ArrowRight' || event.key === 'PageDown' || event.key === ' ') { event.preventDefault(); show(current + 1); }
      else if (event.key === 'ArrowLeft' || event.key === 'PageUp') { event.preventDefault(); show(current - 1); }
      else if (event.key === 'Home') show(0);
      else if (event.key === 'End') show(slides.length - 1);
    });
    // Deep links must work while presenting, not only on first load.
    window.addEventListener('hashchange', () => {
      const target = slides.findIndex(s => '#' + s.id === location.hash);
      if (target >= 0 && target !== current) show(target);
    });
    show(fromHash >= 0 ? fromHash : 0);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
