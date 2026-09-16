'use strict';
let rehearsal = {events:[]}, hostedThermal = {events:[]}, importedThermal = {events:[]};
let thermalCursor = 0, thermalTimer = null, hostedSignature = '';
const facilityNames = {'facility-a':'Facility A','facility-b':'Facility B','facility-c':'Facility C'};
const profileColors = {'facility-a':'#c2f59d','facility-b':'#e8bd80','facility-c':'#9bc9e6'};
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
  $('thermal-mode').textContent=source==='local'?'Local rehearsal · no LLM':source==='hosted'?'Hosted run · recorded evidence':'Imported report · recorded evidence';
  $('thermal-demo').disabled=source!=='local';$('release-button').disabled=source!=='local';
  $('thermal-progress').textContent=`${thermalCursor} / ${record.events.length}`;
  const profiles=[];
  $('thermal-cards').innerHTML=Object.entries(facilityNames).map(([site,label])=>{
    const own=events.filter(e=>e.site===site),released=own.filter(e=>e.status==='released');
    const context=released.find(e=>e.evidence==='context'),balance=released.find(e=>e.evidence==='balance'),audit=released.find(e=>e.evidence==='source_check');
    const spent=Math.max(0,...own.map(e=>Number(e.spent)||0));
    let summary='No evidence released yet.',state='AWAITING EVIDENCE';
    if(context){state=context.finding.gradient_quality==='insufficient'?'ANALOGY REJECTED':'CONTEXT RELEASED';summary=context.finding.gradient_quality==='insufficient'?'The profile fails this demo’s gradient criterion. Further release is denied.':'Context released. Heating delivery and transport remain competing explanations.';}
    if(balance){state='BALANCE RELEASED';summary=`Apparent transport: ${balance.finding.apparent_transport}. Delivered source is not verified.`;profiles.push({site,values:balance.finding.profile_shape,radii:balance.finding.profile_radius});}
    if(audit){state='SOURCE CHECK RELEASED';summary=audit.finding.independent_audit_available?`Independent audit: ${audit.finding.finding}. Corrected transport: ${audit.finding.corrected_transport}.`:'No independent source audit at this facility. Diagnosis remains unresolved.';}
    return `<article class="site-card"><div class="site-top"><span class="site-symbol">${site.slice(-1).toUpperCase()}</span><span class="site-type">${state}</span></div><div class="site-name">${label}</div><p class="thermal-finding">${esc(summary)}</p><div class="budget-meter"><meter min="0" max="5" value="${Math.min(5,spent)}" aria-label="${label} policy units spent"></meter></div><div class="site-foot">${spent} / 5 policy units · ${released.length} release events</div><div class="site-foot">${esc(released.at(-1)?.provenance||'No provenance released yet')}</div></article>`;
  }).join('');
  let svg='<path d="M40 12V200H548" fill="none" stroke="#55664c"/>';
  for(let i=0;i<=4;i++){const y=200-i*45;svg+=`<line x1="40" x2="548" y1="${y}" y2="${y}" stroke="#33402d" stroke-dasharray="3 5"/><text x="30" y="${y+3}" fill="#9baa90" font-size="10" text-anchor="end">${(i/4).toFixed(2)}</text>`;}
  for(const p of profiles){if(!Array.isArray(p.values))continue;const points=p.values.map((v,i)=>{const r=Array.isArray(p.radii)?Number(p.radii[i]):i/(p.values.length-1);return `${40+Math.max(0,Math.min(1,r))*508},${200-Math.max(0,Math.min(1,Number(v)))*180}`;}).join(' ');svg+=`<polyline points="${esc(points)}" fill="none" stroke="${profileColors[p.site]}" stroke-width="3" ${p.site==='facility-b'?'stroke-dasharray="7 4"':''}/>`;}
  svg+='<text x="40" y="221" fill="#9baa90" font-size="10">0</text><text x="525" y="221" fill="#9baa90" font-size="10">1.0 ρ</text>';
  $('thermal-chart').innerHTML=svg;
  $('thermal-legend').textContent=profiles.length?profiles.map(p=>`${facilityNames[p.site]} · ${p.site==='facility-a'?'green':p.site==='facility-b'?'amber dashed':'blue'}`).join(' / '):'No profile has been released.';
  $('thermal-trail').innerHTML=events.map((e,i)=>`<li class="${e.status==='released'?'released':'denied'}"><span class="trace-index">${String(i+1).padStart(2,'0')}</span><div><strong>${esc(facilityNames[e.site]||'Gateway')} · ${esc(e.evidence||'unsupported request')}</strong><small>${esc(e.status||'error')}${e.cached?' · cached, no additional debit':''}${e.reason?' · '+esc(e.reason):''}${e.cost&&!e.cached?' · '+esc(e.cost)+' units':''}</small>${e.evidence_id?'<small>Evidence '+esc(e.evidence_id)+'</small>':''}</div></li>`).join('')||'<li class="muted">No recorded events. Run the local demonstration or choose a hosted report.</li>';
  const a=events.find(e=>e.site==='facility-a'&&e.evidence==='source_check'&&e.status==='released');
  const b=events.find(e=>e.site==='facility-b'&&e.evidence==='source_check'&&e.status==='released');
  $('thermal-conclusion').textContent=a&&b?'The next useful measurement at A: independently calibrated delivered heating. B’s audit supplies a precedent, not A’s diagnosis.':source==='hosted'&&!events.length?'This hosted report contains no merged THERMAL evidence. The earlier toy run is still shown below.':'Releases show what is known so far. Missing or withheld evidence stays unknown.';
  if(thermalCursor>=record.events.length)stopTrail();
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
