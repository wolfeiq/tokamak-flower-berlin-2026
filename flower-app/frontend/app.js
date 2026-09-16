'use strict';
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let currentRun = null;
let renderedReport = null;
let renderedAudit = null;
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json','X-Fusion-UI':'1'},body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function inline(text) { return esc(text).replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>').replace(/`([^`]+)`/g,'<code>$1</code>'); }
function markdown(text) {
  // Deliberately small renderer: raw HTML and model-generated links never execute.
  let list = false;
  const output = [];
  for (const line of text.split('\n')) {
    const bullet = /^\s*[-*]\s+(.+)/.exec(line);
    if (bullet) { if (!list) { output.push('<ul>'); list=true; } output.push(`<li>${inline(bullet[1])}</li>`); continue; }
    if (list) {output.push('</ul>');list=false;}
    const heading = /^(#{1,6})\s+(.+)/.exec(line);
    if (heading) output.push(`<${heading[1].length<=2?'h2':'h3'}>${inline(heading[2])}</${heading[1].length<=2?'h2':'h3'}>`);
    else if (line.trim()) output.push(`<p>${inline(line)}</p>`);
  }
  if (list) output.push('</ul>');
  return output.join('');
}
function showError(id, message) { $(id).textContent=message; $(id).hidden=!message; }
async function loadEvidence() {
  try {
    const data=await api('/api/evidence');
    $('site-cards').innerHTML=data.diagnoses.map((d,i)=>{
      if(d.error) return `<article class="site-card">${esc(d.error)}</article>`;
      const best=d.ranked[0];
      return `<article class="site-card"><div class="site-top"><span class="site-symbol">${['◉','◈','◎'][i]}</span><span class="site-type">${i===0?'REQUESTING FACILITY':'PARTNER FACILITY'}</span></div><div class="site-name">${esc(d.site.replace('demo-','Demo ').toUpperCase())}</div><div class="site-hypothesis">${esc(best.hypothesis.replaceAll('_',' '))}</div><div class="site-facts"><div><strong>${best.efficiency.toFixed(2)}</strong><small>Heating efficiency</small></div><div><strong>${best.loss.toFixed(2)}</strong><small>Heat loss</small></div></div><div class="site-foot">${d.samples} synthetic samples · 16 held out · no raw traces shared</div></article>`;
    }).join('');
  } catch (e) { $('site-cards').textContent=e.message; }
}
async function refreshRun() {
  try {
    const run=await api('/api/run');currentRun=run;
    $('run-status').textContent=run.status.replace('finished:','').replaceAll(':',' · ');
    $('run-button').disabled=run.active;
    $('run-button').innerHTML=run.active?'Investigation running…':'Run investigation <span>↗</span>';
    $('tool-count').textContent=run.audit?.length ?? 0;
    $('run-id').textContent=run.run_id || 'Awaiting submission';
    const link=run.run_id?`https://flower.ai/runs/${encodeURIComponent(run.run_id)}?from=federation&federation=%40marykor%2Fpersonal`:'https://flower.ai/federations/marykor/personal';
    $('run-id').href=link; document.querySelectorAll('.flower-link').forEach(el=>el.href=link);
    $('run-caption').textContent=run.active?'Agent is working · updates automatically':run.status.includes('completed')?'Investigation completed · evidence available':run.status==='idle'?'Ready for an investigation':run.status;
    if(renderedReport!==run.report){$('report-text').innerHTML=markdown(run.report || (run.active?'The hosted agent is investigating. Its report will appear here.':'Submit a brief to start a hosted investigation.'));renderedReport=run.report;}
    $('logs').textContent=run.logs || 'No runtime logs yet.';
    const audit=JSON.stringify(run.audit||[]);
    if(renderedAudit!==audit){
      $('audit').innerHTML=(run.audit||[]).map((call,i)=>`<details><summary>${String(i+1).padStart(2,'0')} · ${esc(call.tool)} ${call.arguments.site?' / '+esc(call.arguments.site):''}</summary><pre>${esc(JSON.stringify(call,null,2))}</pre></details>`).join('');renderedAudit=audit;
    }
    $('audit-count').textContent=`(${run.audit?.length||0} calls)`;
    showError('run-error',run.error||'');
  } catch(e) {showError('run-error',e.message);}
}
function renderChart(result) {
  const maximum=Math.max(...result.baseline.map(x=>x.mean_tracking_error),...result.candidate.map(x=>x.mean_tracking_error),.01)*1.25;
  let svg=''; const top=18,bottom=190,height=bottom-top;
  for(let i=0;i<=4;i++){const y=bottom-height*i/4;svg+=`<line x1="39" x2="563" y1="${y}" y2="${y}" stroke="#34402f" stroke-dasharray="3 5"/><text x="31" y="${y+3}" fill="#8b9c81" font-size="9" text-anchor="end">${(maximum*i/4).toFixed(2)}</text>`;}
  result.baseline.forEach((b,i)=>{
    const center=128+i*174;
    [b,result.candidate[i]].forEach((v,j)=>{const h=v.mean_tracking_error/maximum*height;const x=center+(j===0?-38:5);svg+=`<rect x="${x}" y="${bottom-h}" width="32" height="${Math.max(h,.6)}" rx="3" fill="${j?'#c2f59d':'#64755b'}"/><text x="${x+16}" y="${bottom-h-7}" fill="${j?'#c2f59d':'#98ab8e'}" font-size="10" text-anchor="middle">${v.mean_tracking_error.toFixed(3)}</text>`;});
  });
  $('chart').innerHTML=svg;
  const improved=result.candidate.filter((v,i)=>v.mean_tracking_error<result.baseline[i].mean_tracking_error && v.limit_exceedances===0).length;
  const exceeded=result.candidate.reduce((sum,x)=>sum+x.limit_exceedances,0);
  $('verdict').classList.toggle('pass',result.improves_all_scenarios);
  $('verdict').textContent=`${result.improves_all_scenarios?'✓ Passes toy sensitivity check':'↗ Does not pass all scenarios'} · ${improved}/3 improve · ${exceeded} limit exceedances · ${result.power_scale.toFixed(2)}× at ${result.site}`;
  $('chart').setAttribute('aria-label',`Tracking error for baseline versus candidate. ${result.baseline.map((b,i)=>`Scenario ${i+1}: ${b.mean_tracking_error.toFixed(4)} versus ${result.candidate[i].mean_tracking_error.toFixed(4)}`).join('. ')}`);
}
async function validate() {
  $('validate-button').disabled=true;showError('validation-error','');
  try {renderChart(await api('/api/validate',{site:$('site-select').value,power_scale:Number($('power').value)}));}
  catch(e){showError('validation-error',e.message);}
  finally{$('validate-button').disabled=false;}
}
$('power').addEventListener('input',()=>{$('power-value').textContent=Number($('power').value).toFixed(2)+'×';$('verdict').textContent='Settings changed. Validate to update the result.'; $('verdict').classList.remove('pass');});
$('site-select').addEventListener('change',()=>{$('verdict').textContent='Site changed. Validate to update the result.';$('verdict').classList.remove('pass');});
$('validate-button').addEventListener('click',validate);
$('run-form').addEventListener('submit',async event=>{
  event.preventDefault();$('run-button').disabled=true;showError('run-error','');
  try{await api('/api/run',{prompt:$('prompt').value});await refreshRun();}
  catch(e){showError('run-error',e.message);$('run-button').disabled=false;}
});
$('copy-report').addEventListener('click',async()=>{
  try{await navigator.clipboard.writeText(currentRun?.report||'');$('copy-report').textContent='Copied';setTimeout(()=>$('copy-report').textContent='Copy report',1800);}catch{$('copy-report').textContent='Select report to copy';}
});
loadEvidence();refreshRun();validate();setInterval(refreshRun,4000);
