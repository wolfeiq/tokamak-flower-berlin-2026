// No npm dependencies. Uses a local Chromium browser's DevTools Protocol.
// Run: node presentation/harness/export.mjs
// Optional: CHROME_PATH=/path/to/chrome; --url=http://127.0.0.1:8790
import {spawn} from 'node:child_process';
import {mkdtemp, mkdir, readFile, writeFile, access} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const root=path.dirname(fileURLToPath(import.meta.url));
const candidates=[process.env.CHROME_PATH,'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe','C:/Program Files/Google/Chrome/Application/chrome.exe','/usr/bin/chromium','/usr/bin/google-chrome'].filter(Boolean);
let executable;
for(const candidate of candidates){try{await access(candidate);executable=candidate;break;}catch{}}
if(!executable)throw Error('Set CHROME_PATH to an installed Chromium browser.');
const profile=await mkdtemp(path.join(tmpdir(),'fusion-deck-browser-'));
const browser=spawn(executable,['--headless=new','--disable-extensions','--no-first-run','--no-default-browser-check','--hide-scrollbars','--allow-file-access-from-files','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{windowsHide:true,stdio:['ignore','ignore','pipe']});
const browserUrl=await new Promise((resolve,reject)=>{
  let output='';const timeout=setTimeout(()=>reject(Error('Browser startup timed out.')),20000);
  browser.stderr.on('data',chunk=>{output+=chunk;const match=output.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(match){clearTimeout(timeout);resolve(match[1]);}});
  browser.on('error',reject);
  browser.on('exit',code=>{clearTimeout(timeout);reject(Error(`Browser exited ${code}`));});
});
const address=new URL(browserUrl);
const target=await (await fetch(`http://${address.host}/json/new?about:blank`,{method:'PUT'})).json();
const ws=new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
let nextId=0;const pending=new Map(),errors=[];
ws.addEventListener('message',event=>{const message=JSON.parse(event.data);if(message.id){const pair=pending.get(message.id);if(pair){pending.delete(message.id);message.error?pair.reject(Error(JSON.stringify(message.error))):pair.resolve(message.result);}}else if(message.method==='Runtime.exceptionThrown')errors.push(message.params.exceptionDetails);});
function cdp(method,params={}){return new Promise((resolve,reject)=>{const id=++nextId;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const r=await cdp('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const baseUrl=process.argv.find(arg=>arg.startsWith('--url='))?.slice(6)||pathToFileURL(path.join(root,'index.html')).href;
try{
  await cdp('Page.enable');await cdp('Runtime.enable');
  await cdp('Emulation.setDeviceMetricsOverride',{width:1600,height:900,deviceScaleFactor:1,mobile:false});
  await cdp('Page.navigate',{url:baseUrl});
  for(let i=0;i<100;i++){if(await evaluate('document.readyState === "complete" && !!window.presentation'))break;await sleep(100);}
  await evaluate('document.fonts.ready');
  assert.equal(await evaluate('document.querySelectorAll(".slide").length'),6);
  assert.equal(await evaluate('window.GATEWAY_RECORDING.steps.length'),10);
  assert.equal(await evaluate('window.GATEWAY_RECORDING.steps[4].result.reason'),'analogy-not-applicable');
  assert.equal(await evaluate('window.GATEWAY_RECORDING.steps[9].result.cached'),true);
  await evaluate('window.presentation.show(4)');
  assert.equal(await evaluate('document.getElementById("replay-view").hidden'),true);
  await evaluate('document.getElementById("replay-toggle").click()');
  assert.equal(await evaluate('document.getElementById("walkthrough").hidden'),true);
  await evaluate('window.presentation.advance()');
  assert.equal(await evaluate('window.presentation.state.cursor'),1);
  assert.match(await evaluate('document.getElementById("decision-title").textContent'),/context/);
  for(let i=1;i<10;i++)await evaluate('window.presentation.advance()');
  assert.match(await evaluate('document.getElementById("decision-title").textContent'),/No second charge/);
  assert.equal(await evaluate('document.querySelectorAll("#profile-chart polyline").length'),2);
  assert.match(await evaluate('document.querySelector(".site-C .facility-status").textContent'),/DENIED/);
  await evaluate('document.getElementById("demo-rewind").click()');
  assert.equal(await evaluate('window.presentation.state.cursor'),0);
  assert.equal(await evaluate('document.querySelectorAll("#profile-chart polyline").length'),0);
  await evaluate('document.querySelectorAll("[data-event]")[6].click()');
  assert.equal(await evaluate('window.presentation.state.cursor'),7);
  assert.match(await evaluate('document.getElementById("decision-title").textContent'),/audit/);
  await evaluate('document.getElementById("notes-toggle").click()');
  if(baseUrl.startsWith('http')){
    for(let i=0;i<30;i++){if(await evaluate('window.presentation.state.liveAvailable'))break;await sleep(100);}
    assert.equal(await evaluate('window.presentation.state.liveAvailable'),true);
    await evaluate('document.getElementById("demo-mode").value="live";document.getElementById("demo-mode").dispatchEvent(new Event("change"))');
    for(let i=0;i<10;i++)await evaluate('window.presentation.advance()');
    assert.equal(await evaluate('window.presentation.state.liveNext'),10);
    assert.equal(await evaluate('window.presentation.state.events[4].result.reason'),'analogy-not-applicable');
    assert.equal(await evaluate('window.presentation.state.events[6].result.finding.corrected_transport'),'near-reference');
    assert.equal(await evaluate('window.presentation.state.events[9].result.cached'),true);
    assert.match(await evaluate('document.getElementById("decision-meta").textContent'),/LIVE PYTHON/);
    // Rewind only changes the view; the original ledger debit stays five.
    await evaluate('document.getElementById("demo-rewind").click()');
    assert.equal(await evaluate('window.presentation.state.events[9].result.spent'),5);
    await evaluate('document.getElementById("custom-toggle").click();document.getElementById("request-site").value="C";document.getElementById("request-kind").value="balance";document.getElementById("custom-form").requestSubmit()');
    for(let i=0;i<100;i++){if(!await evaluate('window.presentation.state.busy'))break;await sleep(50);}
    assert.equal(await evaluate('window.presentation.state.events.at(-1).result.reason'),'analogy-not-applicable');
    await evaluate('document.getElementById("demo-mode").value="recorded";document.getElementById("demo-mode").dispatchEvent(new Event("change"))');
    console.log('PASS: live browser calls, source correction, denial, custom request and retained spending.');
  }
  assert.equal(await evaluate('document.getElementById("notes-panel").hidden'),false);
  await evaluate('document.getElementById("notes-toggle").click()');
  // Complete demo in screenshots and PDF. Replay stays interactive on reopening.
  await evaluate('window.presentation.preparePrint()');
  for(let index=0;index<6;index++){
    await evaluate(`window.presentation.show(${index})`);await sleep(150);
    const capture=await cdp('Page.captureScreenshot',{format:'png'});
    await writeFile(path.join(root,`preview-${String(index+1).padStart(2,'0')}.png`),Buffer.from(capture.data,'base64'));
  }
  // Inspect each displayed slide; hidden-slide rectangles cannot establish fit.
  for(let index=0;index<6;index++){
    await evaluate(`window.presentation.show(${index})`);await sleep(220);
    const overflow=await evaluate(`Array.from(document.querySelector('.slide.active').querySelectorAll('h1,h2,h3,.bottom-note,.plain-card,.choice-columns,.rule-strip,.runtime-banner,.run-flow,.return-path,.runtime-bottom,.case-rows,.example-result,.evidence-caption,.evaluation-plan')).filter(el=>el.getClientRects().length && el.getBoundingClientRect().width>0).filter(el=>{const r=el.getBoundingClientRect();return r.right>1520||r.bottom>809||r.left<80||(el.matches('.plain-card')&&el.scrollHeight>el.clientHeight+2);}).map(el=>({element:el.className||el.tagName,bounds:el.getBoundingClientRect().toJSON()}))`);
    assert.deepEqual(overflow,[],`Slide ${index+1} overflow: ${JSON.stringify(overflow)}`);
  }
  const speech=await readFile(path.join(root,'speech_for_harness.txt'),'utf8');
  const sections=speech.split(/\n\d+\. [^\n]+\n/).slice(1).map(s=>s.trim());
  const notes=await evaluate('Array.from(document.querySelectorAll(".speaker-note")).map(n=>n.textContent.trim())');
  assert.deepEqual(notes,sections,'Browser notes must match the speech exactly.');
  assert.equal(await evaluate('document.getElementById("replay-view").hidden'),true);
  // Verify fitting at typical laptop and projector sizes, without changing layout.
  for(const [width,height] of [[1366,768],[1920,1080],[1024,768]]){
    await cdp('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});await sleep(60);
    const bounds=await evaluate('document.getElementById("stage").getBoundingClientRect().toJSON()');
    assert.ok(bounds.left>=-1&&bounds.top>=-1&&bounds.right<=width+1&&bounds.bottom<=height+1);
  }
  await cdp('Emulation.setDeviceMetricsOverride',{width:1600,height:900,deviceScaleFactor:1,mobile:false});
  await evaluate('window.presentation.preparePrint()');
  const pdf=await cdp('Page.printToPDF',{printBackground:true,preferCSSPageSize:true,displayHeaderFooter:false,marginTop:0,marginBottom:0,marginLeft:0,marginRight:0});
  await writeFile(path.join(root,'presentation.pdf'),Buffer.from(pdf.data,'base64'));
  const pdfText=(await readFile(path.join(root,'presentation.pdf'))).toString('latin1');
  assert.equal((pdfText.match(/\/Type\s*\/Page\b/g)||[]).length,6,'PDF must have six pages.');
  if(process.argv.includes('--powerpoint')){
    const directory=path.join(root,'.exports');await mkdir(directory,{recursive:true});
    await evaluate(`(()=>{const style=document.createElement('style');style.textContent='.slide{transition:none!important}.slide-dots,.footer-controls,.demo-controls,.replay-toggle{visibility:hidden!important}.demo-mode select{appearance:none;pointer-events:none}.runtime-reference{pointer-events:none}';document.head.appendChild(style);})()`);
    await cdp('Emulation.setDeviceMetricsOverride',{width:1600,height:900,deviceScaleFactor:1.5,mobile:false});
    for(let index=0;index<6;index++){
      await evaluate(`window.presentation.show(${index})`);
      await sleep(80);
      const capture=await cdp('Page.captureScreenshot',{format:'png'});
      await writeFile(path.join(directory,`slide-${String(index+1).padStart(2,'0')}.png`),Buffer.from(capture.data,'base64'));
    }
    console.log('Exported six 2400×1350 slide images for PowerPoint; worked example is static.');
  }
  assert.deepEqual(errors,[],`Browser errors: ${JSON.stringify(errors)}`);
  console.log('PASS: 6 slides, 10 gateway events, playback, profile disclosure, rewind, notes, 3 viewport sizes, zero browser exceptions.');
  console.log('Exported presentation.pdf and preview-01.png through preview-06.png.');
}finally{
  try{await cdp('Browser.close');}catch{}
  ws.close();browser.kill();
  // Chromium owns its temporary profile. Leave it to the OS temp cleanup;
  // never recursively remove a browser profile while its process may be alive.
}
