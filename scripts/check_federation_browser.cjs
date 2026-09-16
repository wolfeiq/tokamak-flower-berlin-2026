// Browser smoke test. Pass a Python-generated scene HTML path as argv[2].
// With no argument, use a synthetic payload to check rendering/interactions only.
const fs=require('fs'), path=require('path'), os=require('os'), vm=require('vm');
const {spawn}=require('child_process');
const assert=require('assert/strict');
const {pathToFileURL}=require('url');
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'federation-check-'));
let htmlPath=process.argv[2];
if(!htmlPath) {
  const source=fs.readFileSync('viz/federation_scene.py','utf8');
  new vm.Script(source.slice(source.indexOf('const DATA ='),source.lastIndexOf('</script>')));
  const clusters=['thermal','particle','current'].map((key,i)=>({key,label:key,color:['#ff9955','#55ff99','#5599ff'][i],payloadBytes:776,aggregated:true}));
  const devices=[['iter_like',43.7,5.8,-5,-6,6.2,2,1.7],['sparc_like',42.5,-71.6,2,6,1.85,0.57,1.97],['diiid_like',32.9,-117.2,-4,-5,1.67,0.67,1.8],['tcv_like',46.5,6.6,6,5,0.88,0.25,1.7]].map(([key,lat,lon,offsetLon,offsetLat,R,a,kappa],i)=>({key,label:key,badge:'fixture',place:key,lat,lon,offsetLon,offsetLat,R,a,kappa,epsilon:a/R,scale:Math.pow(R/3,0.45),color:'#88bbff',distance:i,B0:5,Ip:1,isTarget:i===0,state:{rho_star:0.01,nu_star:0.1,beta_N:1,q95:3},coils:clusters.map(c=>({...c,cluster:c.key,agents:[{name:'agent',available:true}],weight:0.25,admissible:true,excluded:false}))}));
  const participation={};
  for(let mask=0;mask<16;mask++) {
    const bits=devices.map((d,i)=>(mask>>i)&1), n=bits.reduce((a,b)=>a+b,0);
    participation[bits.join('')]={clusters:Object.fromEntries(clusters.map(c=>[c.key,{aggregated:n>0,effectivePeers:n}])),devices:Object.fromEntries(devices.map((d,i)=>[d.key,Object.fromEntries(clusters.map(c=>[c.key,{weight:bits[i]?1/n:0,admissible:!!bits[i],excluded:!bits[i]}]))]))};
  }
  const payload={clusters,devices,participation,deviceOrder:devices.map(d=>d.key),sending:[],server:{lat:72,lon:-42},target:devices[0].key,bandwidth:1,rule:'mean',round:1,useSimilarity:true};
  const world=fs.readFileSync('viz/world_outline.py','utf8').split(' = (')[1];
  const land=vm.runInNewContext('['+world.replaceAll('(', '[').replaceAll(')', ']'));
  let html=source.split('_TEMPLATE = r"""')[1].split('"""')[0];
  html=html.replace('__SCENE_DATA__',JSON.stringify(payload)).replace('__WORLD_LAND__',JSON.stringify(land)).replace('__THREE_CDN__','https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js').replace('__ORBIT_CDN__','https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js');
  htmlPath=path.join(tmp,'scene.html');fs.writeFileSync(htmlPath,html);
}
const executable=process.env.CHROME_PATH || (process.platform==='win32'?'C:/Program Files/Google/Chrome/Application/chrome.exe':'chromium');
const chrome=spawn(executable,['--headless','--enable-unsafe-swiftshader','--remote-debugging-port=0','--user-data-dir='+path.join(tmp,'profile'),'--window-size='+(process.env.CHECK_VIEWPORT||'1440,900'),'about:blank'],{windowsHide:true,stdio:['ignore','ignore','pipe']});
let ws;const pending=new Map();let id=0;const errors=[];
const timeout=setTimeout(()=>{console.error('Browser check timed out');chrome.kill();process.exit(1);},60000);
(async()=>{
  const endpoint=await new Promise((resolve,reject)=>{let log='';chrome.stderr.on('data',b=>{log+=b;const m=log.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(m)resolve(m[1]);});chrome.on('error',reject);});
  ws=new WebSocket(endpoint);await new Promise(r=>ws.onopen=r);
  ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);if(m.error)p.reject(Error(JSON.stringify(m.error)));else p.resolve(m.result);}if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);};
  function call(method,params={},sessionId){return new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});ws.send(JSON.stringify({id:n,method,params,sessionId}));});}
  const {targetId}=await call('Target.createTarget',{url:'about:blank'});
  const {sessionId}=await call('Target.attachToTarget',{targetId,flatten:true});
  const cmd=(m,p)=>call(m,p,sessionId);
  await cmd('Runtime.enable');await cmd('Page.enable');
  await cmd('Page.navigate',{url:pathToFileURL(path.resolve(htmlPath)).href});
  const evaluate=async expression=>{const r=await cmd('Runtime.evaluate',{expression,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  for(let i=0;i<100;i++){if(await evaluate('document.getElementById("send-status")?.textContent?.includes("sending")'))break;await new Promise(r=>setTimeout(r,300));}
  assert.deepEqual(errors,[]);
  assert.equal(await evaluate('sending.size'),0);assert.equal(await evaluate('links.length'),0);
  async function click(i,drag=false) {
    const point=await evaluate(`(()=>{scene.updateMatrixWorld(true);const p=pickTargets[${i}].getWorldPosition(new THREE.Vector3()).project(camera);const r=renderer.domElement.getBoundingClientRect();return {x:r.left+(p.x+1)*r.width/2,y:r.top+(1-p.y)*r.height/2};})()`);
    await cmd('Input.dispatchMouseEvent',{type:'mousePressed',...point,button:'left',clickCount:1});
    if(drag) await cmd('Input.dispatchMouseEvent',{type:'mouseMoved',x:point.x+20,y:point.y,button:'left',buttons:1});
    await cmd('Input.dispatchMouseEvent',{type:'mouseReleased',...point,button:'left',clickCount:1});
  }
  await click(1);assert.equal(await evaluate('sending.size'),1);assert.ok(await evaluate('links.some(l=>l.dir<0)'));
  await click(2);assert.equal(await evaluate('sending.size'),2);
  await click(1);assert.equal(await evaluate('sending.size'),1);
  await click(2,true);assert.equal(await evaluate('sending.size'),1);
  await click(2);assert.equal(await evaluate('sending.size'),0);assert.equal(await evaluate('links.length'),0);
  assert.deepEqual(errors,[]);
  await click(0);await click(1);await click(2);await click(3);
  assert.equal(await evaluate('sending.size'),4,'Every machine must be clickable without overlays covering it');
  assert.ok(await evaluate('links.every(l=>l.cloud.geometry.attributes.position.array.every(Number.isFinite))'));
  await evaluate('document.getElementById("btn-flow").click()');
  assert.ok(await evaluate('links.every(l=>!l.cloud.visible)'));
  await evaluate('document.getElementById("btn-flow").click()');
  assert.ok(await evaluate('links.every(l=>l.cloud.visible)'));
  await cmd('Input.dispatchMouseEvent',{type:'mouseMoved',x:700,y:700});
  await new Promise(r=>setTimeout(r,400));
  const shot=await cmd('Page.captureScreenshot',{format:'png'});
  fs.writeFileSync(path.join(tmp,'scene.png'),Buffer.from(shot.data,'base64'));
  console.log('PASS: renders, starts silent, multi-selects, stops, ignores drags, returns aggregate to silent target; no JS exceptions.');
  console.log('Screenshot: '+path.join(tmp,'scene.png'));
})().catch(e=>{console.error(e);process.exitCode=1;}).finally(()=>{clearTimeout(timeout);if(ws)ws.close();chrome.kill();});
