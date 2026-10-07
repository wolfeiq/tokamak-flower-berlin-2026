'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const slides = [...document.querySelectorAll('.slide')];
  const demoIndex = slides.findIndex(slide => slide.id === 'investigation');
  const pad = number => String(number).padStart(2, '0');
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let current = 0;
  function resize() {
    const scale = Math.min(innerWidth / 1600, innerHeight / 900);
    $('stage').style.transform = `translate(-50%, -50%) scale(${scale})`;
  }
  function show(index) {
    current = Math.max(0, Math.min(slides.length - 1, index));
    slides.forEach((slide, i) => {
      slide.classList.toggle('active', i === current);
      slide.inert = i !== current;
      slide.setAttribute('aria-hidden', String(i !== current));
    });
    document.querySelectorAll('[data-slide]').forEach(button => button.setAttribute('aria-current', String(Number(button.dataset.slide) === current)));
    $('slide-counter').textContent = `${pad(current + 1)} / ${pad(slides.length)}`;
    $('progress-fill').style.width = `${(current + 1) / slides.length * 100}%`;
    $('previous').disabled = current === 0;
    $('next').disabled = current === slides.length - 1;
    $('notes-text').textContent = slides[current].querySelector('.speaker-note').textContent;
    $('footer-label').textContent = current === demoIndex ? 'SYNTHETIC REHEARSAL · NO MODEL CALLS' : 'FUSION INVESTIGATOR · RESEARCH PROTOTYPE';
    history.replaceState(null, '', `#${slides[current].id}`);
    if (current !== demoIndex) pause();
    if (current === 0) drawPlasma(performance.now());
  }
  function toggleNotes() { $('notes-panel').hidden = !$('notes-panel').hidden; }
  async function fullscreen() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await document.documentElement.requestFullscreen();
    } catch { /* Browser may decline fullscreen; deck remains usable. */ }
  }
  $('previous').addEventListener('click', () => show(current - 1));
  $('next').addEventListener('click', () => show(current + 1));
  $('notes-toggle').addEventListener('click', toggleNotes);
  $('fullscreen').addEventListener('click', fullscreen);
  document.querySelectorAll('[data-slide]').forEach(button => button.addEventListener('click', () => show(Number(button.dataset.slide))));
  addEventListener('resize', resize);
  addEventListener('hashchange', () => {
    const index = slides.findIndex(slide => `#${slide.id}` === location.hash);
    if (index >= 0) show(index);
  });
  document.addEventListener('keydown', event => {
    if (/^(INPUT|TEXTAREA|SELECT|BUTTON)$/.test(event.target.tagName) || event.ctrlKey || event.metaKey || event.altKey) return;
    if (['ArrowRight', 'PageDown', ' '].includes(event.key)) { event.preventDefault(); show(current + 1); }
    else if (['ArrowLeft', 'PageUp'].includes(event.key)) { event.preventDefault(); show(current - 1); }
    else if (event.key === 'Home') show(0);
    else if (event.key === 'End') show(slides.length - 1);
    else if (event.key.toLowerCase() === 'f') fullscreen();
    else if (event.key.toLowerCase() === 'n') toggleNotes();
    else if (event.key.toLowerCase() === 'd' && current === demoIndex) { event.preventDefault(); setReplay(true); pause(); advance(); }
    else if (event.key === 'Escape') { $('notes-panel').hidden = true; $('custom-form').hidden = true; }
  });

  // The offline interaction replays captured Python results. It never invents
  // findings or implements a second version of the disclosure policy in JS.
  const recording = window.GATEWAY_RECORDING;
  const recorded = recording?.steps || [];
  function setReplay(open) {
    pause();
    $('walkthrough').hidden = open;
    $('replay-view').hidden = !open;
    $('investigation').classList.toggle('replaying', open);
    $('replay-toggle').setAttribute('aria-expanded', String(open));
    $('replay-toggle').textContent = open ? 'Back to worked example' : 'Inspect gateway replay';
  }
  $('replay-toggle').addEventListener('click', () => setReplay($('replay-view').hidden));
  let mode = 'recorded', cursor = 0, liveEvents = [], liveNext = 0;
  let timer = null, playing = false, busy = false, liveAvailable = false;
  const labels = {context:'Context', balance:'Balance', source_check:'Source', raw_logs:'Raw logs'};
  const devices = {A:'DIII-D-like', B:'SPARC-like', C:'TCV-like'};
  const colors = {A:'#ff986e', B:'#8ce7db', C:'#9aafff'};
  const events = () => mode === 'recorded' ? recorded : liveEvents;
  const visible = () => events().slice(0, cursor);
  function pause() {
    clearTimeout(timer); timer = null; playing = false;
    $('demo-play').innerHTML = '<span>▶</span> Play sequence';
  }
  function error(message) { $('demo-error').textContent = message; $('demo-error').hidden = !message; }
  function complete() { return cursor >= events().length && (mode === 'recorded' || liveNext >= recorded.length); }
  async function advance() {
    if (busy) return;
    error('');
    if (cursor < events().length) { cursor++; render(); return; }
    if (mode !== 'live' || liveNext >= recorded.length) { pause(); return; }
    busy = true; render();
    try {
      const response = await fetch('/api/request', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(recorded[liveNext].request), signal:AbortSignal.timeout(15000)});
      const step = await response.json();
      if (!response.ok) throw Error(step.error || 'Gateway request failed.');
      liveEvents.push(step); liveNext++; cursor = liveEvents.length;
    } catch (exception) { error(`Live request failed: ${exception.message} Select recorded evidence to continue offline.`); pause(); }
    finally { busy = false; render(); }
  }
  async function play() {
    if (playing) { pause(); return; }
    if (complete()) cursor = 0;
    playing = true;
    $('demo-play').innerHTML = '<span>Ⅱ</span> Pause sequence';
    const tick = async () => {
      if (!playing) return;
      await advance();
      if (complete()) { pause(); return; }
      if (playing) timer = setTimeout(tick, 3500);
    };
    await tick();
  }
  function describe(step) {
    const {request, result:r} = step, site = request.site, kind = request.kind;
    if (r.cached) return ['CACHED RELEASE', 'Same finding. No second charge.', `Facility ${site} returns its previously released ${labels[kind].toLowerCase()} evidence. Spending remains ${r.spent} / 5.`, 'cached'];
    if (r.reason === 'analogy-not-applicable') return ['ANALOGY-NOT-APPLICABLE', 'C’s balance request is denied.', 'Its temperature gradient fails this demo’s applicability criterion. No balance profile is released.', 'denied'];
    if (r.reason === 'unsupported-request') return ['UNSUPPORTED-REQUEST', 'Raw records stay behind the gateway.', 'The fixed evidence catalogue has no raw-log product. The request is refused in code.', 'denied'];
    if (r.reason === 'prerequisite-missing') return ['PREREQUISITE-MISSING', 'The earlier release is required.', 'Context precedes balance; balance precedes source check. The gateway refuses to skip the chain.', 'denied'];
    if (r.status !== 'released') return ['REQUEST DENIED', 'The gateway withheld this product.', r.reason || 'No evidence was released.', 'denied'];
    if (kind === 'context' && r.finding.gradient_quality === 'insufficient') return ['CONTEXT RELEASED · 1 UNIT', 'C’s gradient is insufficient.', 'The barely heated synthetic case fails the configured threshold. Next, request its balance evidence.', ''];
    if (kind === 'context') return ['CONTEXT RELEASED · 1 UNIT', `${site}’s context is released.`, 'The solved case has verified steady state and a usable gradient. The source-versus-transport question remains open.', ''];
    if (kind === 'balance') return ['BALANCE RELEASED · 2 UNITS', 'Transport appears above reference.', `Facility ${site} releases a coarse band and eleven rounded profile samples. Its source is still marked unverified.`, ''];
    if (kind === 'source_check' && r.finding.independent_audit_available) return ['SOURCE CHECK RELEASED · 2 UNITS', 'B’s audit changes the interpretation.', 'Delivered power is below assumed power. Recomputed transport is near reference. This suggests what A should measure.', ''];
    return ['SOURCE CHECK RELEASED · 2 UNITS', 'A still needs a measurement.', 'No independent source audit is available at A. Its cause remains unresolved; B’s result provides a useful precedent.', ''];
  }
  function renderCards(shown) {
    const active = shown.at(-1)?.request.site;
    $('facility-cards').innerHTML = ['A','B','C'].map(site => {
      const own = shown.filter(step => step.request.site === site);
      const released = own.filter(step => step.result.status === 'released');
      const context = released.find(step => step.request.kind === 'context')?.result;
      const balance = released.find(step => step.request.kind === 'balance')?.result;
      const source = released.find(step => step.request.kind === 'source_check')?.result;
      const denied = own.some(step => step.result.reason === 'analogy-not-applicable');
      const spent = Math.max(0, ...own.map(step => Number(step.result.spent) || 0));
      let state = 'AWAITING EVIDENCE', finding = 'No findings released yet.';
      if (context) { state = 'CONTEXT RELEASED'; finding = 'Steady state verified.<br>Gradient is usable.'; }
      if (context?.finding.gradient_quality === 'insufficient') { state = 'GRADIENT INSUFFICIENT'; finding = 'Barely heated case.<br>Gradient criterion fails.'; }
      if (balance) { state = 'BALANCE RELEASED'; finding = '<strong>Above-reference transport.</strong><br>Heating source is unverified.'; }
      if (source) {
        state = source.finding.independent_audit_available ? 'AUDIT AVAILABLE' : 'CAUSE UNRESOLVED';
        finding = source.finding.independent_audit_available ? '<strong>Corrected: near reference.</strong><br>Delivered power below assumed.' : '<strong>No independent power audit.</strong><br>Next step: measure delivered heat.';
      }
      if (denied) { state = 'BALANCE DENIED'; finding = '<strong>Analogy not applicable.</strong><br>No balance profile released.'; }
      return `<article class="facility-card site-${site} ${active === site ? 'focused' : ''}"><div class="facility-top"><span class="facility-id">${site}</span><div><div class="facility-name">Facility ${site}</div><div class="facility-device">${devices[site]} · synthetic</div></div><span class="facility-status">${state}</span></div><p class="facility-finding">${finding}</p><div class="facility-budget"><div class="budget-bars" aria-hidden="true">${Array.from({length:5},(_,i)=>`<i class="${i<spent?'spent':''}"></i>`).join('')}</div><span class="budget-count">${spent} / 5 units spent</span></div></article>`;
    }).join('');
  }
  function renderChart(shown) {
    const releases = ['A','B'].map(site => shown.find(step => step.request.site === site && step.request.kind === 'balance' && step.result.status === 'released')).filter(Boolean);
    let markup = '<defs><linearGradient id="profile-wash" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#ff986e" stop-opacity=".10"/><stop offset="1" stop-color="#ff986e" stop-opacity="0"/></linearGradient></defs>';
    for (let i=0; i<=4; i++) {
      const y = 179-i*38;
      markup += `<path d="M43 ${y}H630" stroke="#26333f" stroke-dasharray="3 6"/><text x="31" y="${y+4}" text-anchor="end">${(i/4).toFixed(2)}</text>`;
    }
    markup += '<path d="M43 20V179H630" stroke="#465565" fill="none"/><text x="43" y="201">0</text><text x="322" y="201">0.5</text><text x="602" y="201">1.0</text>';
    if (!releases.length) markup += '<text class="empty-label" x="336" y="100" text-anchor="middle">Profiles appear only after a balance release.</text>';
    for (const step of releases) {
      const f = step.result.finding, color = colors[step.request.site];
      const points = f.profile_shape.map((value,i)=>[43+f.profile_radius[i]*587,179-value*152]);
      if (step.request.site === 'A') markup += `<path d="M${points[0][0]} 179 L${points.map(p=>p.join(' ')).join(' L')} L${points.at(-1)[0]} 179 Z" fill="url(#profile-wash)"/>`;
      markup += `<polyline points="${points.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${color}" stroke-width="2.5" ${step.request.site==='B'?'stroke-dasharray="7 5"':''}/>`;
      markup += points.map(([x,y])=>`<circle cx="${x}" cy="${y}" r="3" fill="${color}" stroke="#080d15" stroke-width="1"/>`).join('');
    }
    $('profile-chart').innerHTML = markup;
  }
  function render() {
    const shown = visible(), last = shown.at(-1);
    renderCards(shown); renderChart(shown);
    const description = last ? describe(last) : ['READY TO INVESTIGATE', 'Begin with A’s context.', 'Step through ten gateway decisions, from the first release to a cached repeat.', ''];
    $('decision-kicker').textContent = description[0];
    $('decision-title').textContent = description[1];
    $('decision-detail').textContent = description[2];
    $('decision-panel').className = `decision-panel ${description[3]}`;
    $('decision-meta').textContent = last ? `${last.result.evidence_id ? 'EVIDENCE '+last.result.evidence_id : 'DECISION '+(last.result.reason || last.result.status)} · ${mode==='recorded'?'RECORDED':'LIVE PYTHON'}\n${last.result.provenance || last.result.case_id || ''}` : 'A real evidence trail. Every result is inspectable.';
    const timeline = mode === 'recorded' ? recorded : liveEvents.length > recorded.length ? liveEvents : [...liveEvents, ...recorded.slice(liveEvents.length)];
    $('demo-timeline').innerHTML = timeline.map((step,i) => `<button class="event-tick ${i<cursor?'done':''} ${i===cursor-1?'selected':''} ${step.result.status==='denied'?'denied':''} ${step.result.cached?'cached':''}" data-event="${i}" aria-label="Event ${i+1}: facility ${escape(step.request.site)}, ${escape(labels[step.request.kind])}" ${i>=events().length || busy?'disabled':''}><span class="tick-number">${pad(i+1)}</span>${escape(step.request.site)} · ${escape(labels[step.request.kind])}</button>`).join('');
    $('event-counter').textContent = `${pad(cursor)} / ${pad(Math.max(recorded.length, events().length))} EVENTS`;
    $('demo-next').disabled = busy || complete();
    $('demo-play').disabled = busy || !recorded.length;
    $('demo-rewind').disabled = busy;
    $('demo-mode').disabled = busy;
    $('custom-toggle').hidden = mode !== 'live';
    $('connection-caption').textContent = mode === 'recorded' ? 'Captured Python outputs · works offline' : 'Actual Python gateway · ledger retained across restarts';
    if (complete() && playing) pause();
  }
  $('demo-play').addEventListener('click', play);
  $('demo-next').addEventListener('click', () => {pause(); advance();});
  $('demo-rewind').addEventListener('click', () => {pause(); cursor = 0; render();});
  $('demo-timeline').addEventListener('click', event => {
    const button = event.target.closest('[data-event]');
    if (button && !button.disabled) {pause(); cursor = Number(button.dataset.event)+1; render();}
  });
  $('demo-mode').addEventListener('change', () => {
    pause(); mode = $('demo-mode').value; cursor = 0; error(''); $('custom-form').hidden = true; render();
  });
  $('custom-toggle').addEventListener('click', () => {pause(); $('custom-form').hidden = !$('custom-form').hidden;});
  $('custom-form').addEventListener('submit', async event => {
    event.preventDefault(); if (busy || !liveAvailable || mode !== 'live') return;
    busy = true; render(); error('');
    try {
      const response = await fetch('/api/request', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({site:$('request-site').value,kind:$('request-kind').value}), signal:AbortSignal.timeout(15000)});
      const step = await response.json();
      if (!response.ok) throw Error(step.error || 'Gateway request failed.');
      liveEvents.push(step); cursor = liveEvents.length; $('custom-form').hidden = true;
    } catch (exception) {error(`Live request failed: ${exception.message}`);}
    finally {busy=false; render();}
  });
  async function detectLive() {
    if (!/^https?:$/.test(location.protocol)) return;
    try {
      const response = await fetch('/api/status', {signal:AbortSignal.timeout(2000)});
      if (!response.ok || (await response.json()).mode !== 'live-python-gateway') return;
      liveAvailable = true;
      $('demo-mode').querySelector('[value=live]').disabled = false;
    } catch { /* Bundled replay is fully functional without a backend. */ }
  }

  function drawPlasma() {} // Kept for existing export callers; the talk uses a static diagram.
  function preparePrint() {pause();setReplay(false);mode='recorded';$('demo-mode').value=mode;cursor=recorded.length;render();$('notes-panel').hidden=true;document.documentElement.classList.add('exporting');slides.forEach(slide=>{slide.inert=false;slide.removeAttribute('aria-hidden');});}
  addEventListener('beforeprint', preparePrint);
  addEventListener('afterprint', () => {document.documentElement.classList.remove('exporting');show(current);});
  window.presentation = {show,advance,preparePrint,render,drawPlasma,setReplay,get state(){return {current,mode,cursor,liveNext,liveAvailable,busy,events:events()};}};
  resize(); render(); detectLive();
  show(Math.max(0,slides.findIndex(slide=>`#${slide.id}`===location.hash)));
  if (!recorded.length) error('The bundled evidence recording is missing. Regenerate it with capture_evidence.py.');
})();
