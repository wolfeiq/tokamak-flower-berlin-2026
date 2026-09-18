'use strict';
let rehearsal = {events:[]}, hostedThermal = {events:[]}, importedThermal = {events:[]};
let thermalCursor = 0, thermalTimer = null, hostedSignature = '';
const facilityNames = {'facility-a':'Facility A','facility-b':'Facility B','facility-c':'Facility C'};
const profileColors = {'facility-a':'#FF5A36','facility-b':'#E9E7E1','facility-c':'#6E86FF'};
function selectFacility(site) {
  if (!facilityNames[site]) return;
  $('release-site').value = site;
  document.querySelectorAll('[data-facility]').forEach(card => {
    const selected = card.dataset.facility === site;
    card.classList.toggle('selected', selected);
    card.querySelector('.facility-select').setAttribute('aria-pressed', String(selected));
    card.querySelector('.facility-select-state').textContent = selected ? 'Selected' : 'Select device →';
  });
  const physics = $('physics-site'), letter = site.slice(-1).toUpperCase();
  if (physics.value !== letter) {
    physics.value = letter;
    physics.dispatchEvent(new Event('change'));
  }
  $('facility-selection').textContent = `${facilityNames[site]} selected for evidence requests and thermal tools.`;
}
function createFacilityCards() {
  $('thermal-cards').innerHTML = Object.entries(facilityNames).map(([site,label]) => `
    <article class="site-card facility-card" data-facility="${site}">
      <div class="site-top"><span class="site-symbol">${site.slice(-1).toUpperCase()}</span><span class="site-type">AWAITING EVIDENCE</span></div>
      <div class="facility-model"><iframe title="${label} interactive 3D tokamak" loading="lazy" hidden></iframe><p class="facility-model-status">Loading reactor geometry…</p></div>
      <button type="button" class="facility-select" aria-pressed="false" aria-label="Select ${label}"><span><strong>${label}</strong><small class="facility-device">Synthetic facility</small></span><span class="facility-select-state">Select device →</span></button>
      <p class="thermal-finding"></p><div class="budget-meter"><meter min="0" max="5" value="0" aria-label="${label} policy units spent"></meter></div><div class="site-foot facility-budget"></div><div class="site-foot facility-provenance"></div>
    </article>`).join('');
  document.querySelectorAll('[data-facility]').forEach(card => {
    card.querySelector('.facility-select').addEventListener('click', () => selectFacility(card.dataset.facility));
  });
  selectFacility($('release-site').value);
}
async function loadFacilityModels() {
  try {
    const devices = await api('/api/devices');
    const labels = {diiid_like:'DIII-D-like', sparc_like:'SPARC-like', tcv_like:'TCV-like'};
    for (const [letter, dev] of Object.entries(devices)) {
      const card = document.querySelector(`[data-facility="facility-${letter.toLowerCase()}"]`);
      if (!card || !labels[dev.name]) continue;
      card.querySelector('.facility-device').textContent = `${labels[dev.name]} · ${dev.B_0} T`;
      const frame = card.querySelector('iframe');
      frame.dataset.device = dev.name;
      // Verify availability before embedding, so a missing asset has a useful fallback.
      try {
        const response = await fetch('/twin/' + dev.name + '.html', {method:'HEAD'});
        if (!response.ok) throw Error('Geometry unavailable');
        frame.src = '/twin/' + dev.name + '.html#selector';
        frame.hidden = false;
        card.querySelector('.facility-model-status').textContent = 'Click to select · drag to rotate';
      } catch {
        card.querySelector('.facility-model-status').textContent = '3D unavailable · select this device below';
      }
    }
  } catch {
    document.querySelectorAll('.facility-model-status').forEach(status => {
      if (status.textContent.startsWith('Loading')) status.textContent = '3D unavailable · select this device below';
    });
  }
}
window.addEventListener('message', event => {
  if (event.origin !== location.origin || event.data?.type !== 'fusion:select-facility') return;
  const card = [...document.querySelectorAll('[data-facility]')].find(item => item.querySelector('iframe').contentWindow === event.source);
  if (card) selectFacility(card.dataset.facility);
});
$('release-site').addEventListener('change', event => selectFacility(event.target.value));
$('physics-site').addEventListener('change', event => selectFacility('facility-' + event.target.value.toLowerCase()));
createFacilityCards();
loadFacilityModels();
function selectedThermal() {return $('thermal-source').value==='hosted'?hostedThermal:$('thermal-source').value==='imported'?importedThermal:rehearsal;}
function stopTrail(){clearInterval(thermalTimer);thermalTimer=null;$('thermal-play').textContent='Play trail';}
function refreshHostedThermal(run) {
  const events=(run.audit||[]).filter(x=>x.tool==='request_evidence').map(x=>x.result);
  const signature=JSON.stringify(events);
  if(signature===hostedSignature)return;
  hostedSignature=signature;hostedThermal={events,mode:run.status,run_id:run.run_id};
  if($('thermal-source').value==='hosted'){thermalCursor=events.length;renderThermal();}
}
function renderThermal(){
  const source=$('thermal-source').value,record=selectedThermal();
  thermalCursor=Math.min(thermalCursor,record.events.length);
  const events=record.events.slice(0,thermalCursor);
  renderEvidenceStory(events, source);
  $('thermal-mode').textContent=source==='local'?'Local rehearsal · no LLM':source==='hosted'?'Hosted run · recorded evidence':'Imported report · recorded evidence';
  $('thermal-demo').disabled=source!=='local';$('release-button').disabled=source!=='local';
  $('thermal-progress').textContent=`${thermalCursor} / ${record.events.length}`;
  const profiles=[];
  Object.entries(facilityNames).forEach(([site,label])=>{
    const own=events.filter(e=>e.site===site),released=own.filter(e=>e.status==='released');
    const context=released.find(e=>e.evidence==='context'),balance=released.find(e=>e.evidence==='balance'),audit=released.find(e=>e.evidence==='source_check');
    const spent=Math.max(0,...own.map(e=>Number(e.spent)||0));
    let summary='No evidence released yet.',state='AWAITING EVIDENCE';
    if(context){state=context.finding.gradient_quality==='insufficient'?'ANALOGY REJECTED':'CONTEXT RELEASED';summary=context.finding.gradient_quality==='insufficient'?'The profile fails this demo’s gradient criterion. Further release is denied.':'Context released. Heating delivery and transport remain competing explanations.';}
    if(balance){state='BALANCE RELEASED';summary=`Apparent transport: ${balance.finding.apparent_transport}. Delivered source is not verified.`;profiles.push({site,values:balance.finding.profile_shape,radii:balance.finding.profile_radius});}
    if(audit){state='SOURCE CHECK RELEASED';summary=audit.finding.independent_audit_available?`Independent audit: ${audit.finding.finding}. Corrected transport: ${audit.finding.corrected_transport}.`:'No independent source audit at this facility. Diagnosis remains unresolved.';}
    const card = document.querySelector(`[data-facility="${site}"]`);
    card.querySelector('.site-type').textContent = state;
    card.querySelector('.thermal-finding').textContent = summary;
    card.querySelector('meter').value = Math.min(5, spent);
    card.querySelector('.facility-budget').textContent = `${spent} / 5 policy units · ${released.length} release events`;
    card.querySelector('.facility-provenance').textContent = released.at(-1)?.provenance || 'No provenance released yet';
  });
  let svg='<path d="M40 12V200H548" fill="none" stroke="#2F333A"/>';
  for(let i=0;i<=4;i++){const y=200-i*45;svg+=`<line x1="40" x2="548" y1="${y}" y2="${y}" stroke="#23262B" stroke-dasharray="3 5"/><text x="30" y="${y+3}" fill="#5C6068" font-size="10" text-anchor="end">${(i/4).toFixed(2)}</text>`;}
  for(const p of profiles){if(!Array.isArray(p.values))continue;const points=p.values.map((v,i)=>{const r=Array.isArray(p.radii)?Number(p.radii[i]):i/(p.values.length-1);return `${40+Math.max(0,Math.min(1,r))*508},${200-Math.max(0,Math.min(1,Number(v)))*180}`;}).join(' ');svg+=`<polyline points="${esc(points)}" fill="none" stroke="${profileColors[p.site]}" stroke-width="3" ${p.site==='facility-b'?'stroke-dasharray="7 4"':''}/>`;}
  svg+='<text x="40" y="221" fill="#5C6068" font-size="10">0</text><text x="525" y="221" fill="#5C6068" font-size="10">1.0 ρ</text>';
  $('thermal-chart').innerHTML=svg;
  $('thermal-legend').textContent=profiles.length?profiles.map(p=>`${facilityNames[p.site]} · ${p.site==='facility-a'?'coral':p.site==='facility-b'?'white dashed':'blue'}`).join(' / '):'No profile has been released.';
  $('thermal-trail').innerHTML=events.map((e,i)=>`<li class="${e.status==='released'?'released':'denied'}"><span class="trace-index">${String(i+1).padStart(2,'0')}</span><div><strong>${esc(facilityNames[e.site]||'Gateway')} · ${esc(e.evidence||'unsupported request')}</strong><small>${esc(e.status||'error')}${e.cached?' · cached, no additional debit':''}${e.reason?' · '+esc(e.reason):''}${e.cost&&!e.cached?' · '+esc(e.cost)+' units':''}</small>${e.evidence_id?'<small>Evidence '+esc(e.evidence_id)+'</small>':''}</div></li>`).join('')||'<li class="muted">No recorded events. Run the local demonstration or choose a hosted report.</li>';
  $('thermal-conclusion').textContent=$('sharing-next-step').textContent;
  if(thermalCursor>=record.events.length)stopTrail();
}
function renderEvidenceStory(events, source) {
  const released = (site, kind) => events.find(e => e.site === site && e.evidence === kind && e.status === 'released');
  const before = released('facility-b', 'balance');
  const audit = released('facility-b', 'source_check');
  const aContext = released('facility-a', 'context');
  const aAudit = released('facility-a', 'source_check');
  const cContext = released('facility-c', 'context');
  const hasAudit = audit?.finding?.independent_audit_available === true;
  const beforeBand = before?.finding?.apparent_transport;
  const afterBand = hasAudit ? audit.finding.corrected_transport : null;
  const bands = {'above-reference':{y:46,label:'Above reference'},'near-reference':{y:111,label:'Near reference'},'below-reference':{y:176,label:'Below reference'}};
  const first = bands[beforeBand], last = bands[afterBand];
  let chart = Object.values(bands).map(b => `<line x1="157" x2="535" y1="${b.y}" y2="${b.y}" stroke="#2f333a" stroke-dasharray="3 5"/><text x="143" y="${b.y+5}" text-anchor="end" fill="#a9adb5" font-size="14">${b.label}</text>`).join('');
  if (first && last) chart += `<path d="M230 ${first.y} L445 ${last.y}" stroke="#adb3be" stroke-width="2" fill="none"/>`;
  for (const [x, band, color] of [[230,first,'#ff5a36'],[445,last,'#75d7bb']]) {
    chart += band ? `<circle cx="${x}" cy="${band.y}" r="8" fill="${color}"/>` : `<text x="${x}" y="116" text-anchor="middle" fill="#a9adb5" font-size="12">Not released</text>`;
  }
  chart += '<text x="230" y="220" text-anchor="middle" fill="#e9e7e1" font-size="14">Before audit</text><text x="445" y="220" text-anchor="middle" fill="#e9e7e1" font-size="14">After audit</text>';
  $('audit-comparison-chart').innerHTML = chart;
  $('audit-comparison-chart').setAttribute('aria-label', `B heat-spreading estimate: before audit ${first?.label || 'not released'}; after audit ${last?.label || 'not released'}. Qualitative categories.`);
  $('sharing-source').textContent = source === 'hosted' ? 'Hosted agent evidence' : source === 'imported' ? 'Imported evidence' : 'Local rehearsal · no LLM';
  const auditSupports = hasAudit && audit.finding.finding === 'delivered-below-assumed' && beforeBand === 'above-reference' && afterBand === 'near-reference';
  $('sharing-audit').textContent = auditSupports
    ? 'B’s audit: less heating arrived than assumed. Using the audited heating brings the heat-spreading estimate back near its reference.'
    : hasAudit ? `B’s released audit: ${audit.finding.finding || 'unspecified'}. Corrected estimate: ${afterBand || 'not released'}.`
    : audit ? 'B has no independent heating audit in this evidence. The cause remains unresolved.'
    : 'B’s audit has not been released at this point. We cannot show a corrected estimate yet.';
  const aMissingAudit = aAudit?.finding?.independent_audit_available === false;
  const aComparable = aContext?.finding?.gradient_quality === 'usable' && aContext?.finding?.symptom === 'weak-temperature-response';
  $('sharing-next-step').textContent = auditSupports && aComparable && aMissingAudit
    ? 'Independently measure how much heating actually reaches A. B’s shared audit gives A a reason to prioritize that check.'
    : auditSupports && aComparable && !aAudit
    ? 'Check whether A has an independent heating audit. B’s finding makes delivered heating worth investigating.'
    : auditSupports && !aComparable
    ? 'B has a useful local finding. Read A’s context before deciding whether it is relevant to A.'
    : auditSupports
    ? 'A has its own audit evidence; assess that directly before transferring B’s explanation.'
    : 'No supported cross-facility recommendation yet. Release the context, balance and audit evidence to follow the investigation.';
  $('sharing-a-status').textContent = aMissingAudit ? 'A’s cause is still unknown. B’s result is a clue for the next measurement.' : aAudit ? 'A’s own audit must determine what can be concluded about A.' : 'A’s independent audit status is not yet established in the released evidence.';
  $('sharing-c-status').textContent = cContext?.finding?.gradient_quality === 'insufficient' ? 'C is excluded: its nearly flat temperature profile fails this demo’s comparison rule.' : 'C’s relevance depends on its released context.';
  $('sharing-evidence-ids').innerHTML = [before,audit,aContext,aAudit,cContext].filter(Boolean).map(e => `<p>${esc(facilityNames[e.site])} · ${esc(e.evidence)} · ${esc(e.evidence_id || 'No evidence ID')}<br><span>${esc(e.provenance || 'Provenance not supplied')}</span></p>`).join('') || '<p>No supporting evidence released yet.</p>';
}
async function loadThermal(){try{rehearsal=await api('/api/thermal');thermalCursor=rehearsal.events.length;renderThermal();}catch(e){showError('thermal-error',e.message);}}
$('thermal-source').addEventListener('change',()=>{stopTrail();thermalCursor=selectedThermal().events.length;renderThermal();});
$('thermal-demo').addEventListener('click',async()=>{stopTrail();$('thermal-demo').disabled=true;showError('thermal-error','');try{rehearsal=await api('/api/replay',{});thermalCursor=rehearsal.events.length;renderThermal();}catch(e){showError('thermal-error',e.message);}finally{$('thermal-demo').disabled=false;}});
$('release-form').addEventListener('submit',async e=>{e.preventDefault();stopTrail();showError('thermal-error','');$('release-button').disabled=true;try{rehearsal=await api('/api/release',{site:$('release-site').value,kind:$('release-kind').value});thermalCursor=rehearsal.events.length;renderThermal();}catch(err){showError('thermal-error',err.message);}finally{$('release-button').disabled=$('thermal-source').value!=='local';}});
$('thermal-next').addEventListener('click',()=>{thermalCursor=Math.min(thermalCursor+1,selectedThermal().events.length);renderThermal();});
$('thermal-rewind').addEventListener('click',()=>{stopTrail();thermalCursor=0;renderThermal();});
$('thermal-play').addEventListener('click',()=>{if(thermalTimer){stopTrail();return;}if(thermalCursor>=selectedThermal().events.length)thermalCursor=0;$('thermal-play').textContent='Pause';thermalTimer=setInterval(()=>{thermalCursor++;renderThermal();},900);});
$('thermal-file').addEventListener('change',async e=>{try{stopTrail();const file=e.target.files[0];if(!file)return;if(file.size>2000000)throw Error('Report exceeds 2 MB');const record=JSON.parse(await file.text());if(!Array.isArray(record.events)||record.events.length>500)throw Error('Invalid report');const events=record.events.map(x=>({...x,site:facilityNames[x.site]?x.site:'facility-'+String(x.site).toLowerCase()}));for(const x of events){if(x.status==='released'&&(!x.finding||typeof x.finding!=='object'))throw Error('Released event is missing its finding');}importedThermal={...record,events};$('thermal-source').value='imported';thermalCursor=events.length;showError('thermal-error','');renderThermal();}catch(err){showError('thermal-error',err.message);}});
loadThermal();
