"""Geographic federation with digital-twin TF coils and selectable senders.

Channel colors on physical coils denote logical agent roles, not magnet wiring.
The server is schematic; site pins are geographic and station sizes compressed.
"""

from __future__ import annotations

import json

from .world_outline import LAND

# The Three.js build the rest of the project already pins (see
# ``assets/3d/html/*.html``). Kept identical on purpose: two Three.js versions
# in one Streamlit app is a debugging afternoon nobody needs.
_THREE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
_ORBIT_CDN = (
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"
)

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  :root {
    --bg: #060d17;
    --panel: rgba(12, 23, 36, 0.78);
    --border: rgba(155, 191, 219, 0.14);
    --text: #b4c8d9;
    --muted: #71899f;
    --bright: #ecf5fc;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { width: 100%; height: 100%; overflow: hidden;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  #stage { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
  #vignette { position: absolute; inset: 0; pointer-events: none; z-index: 1;
    background: radial-gradient(ellipse at 50% 42%, transparent 36%, rgba(3,9,17,.38) 100%); }
  #masthead { position: absolute; top: 25px; left: 28px; z-index: 10; pointer-events: none; }
  .eyebrow { font-size: 9px; letter-spacing: .2em; text-transform: uppercase; color: #86a7bf; font-weight: 600; }
  #masthead h1 { font-size: 24px; font-weight: 500; letter-spacing: -.045em; color: var(--bright); margin: 7px 0; }
  #masthead p { font-size: 11px; color: var(--muted); }
  .panel { position: absolute; z-index: 10; background: var(--panel);
    backdrop-filter: blur(18px); border: 1px solid var(--border);
    box-shadow: 0 12px 38px rgba(0,0,0,.15), inset 0 1px rgba(255,255,255,.025);
    border-radius: 14px; padding: 14px 16px; font-size: 11px; line-height: 1.65; }
  #legend { bottom: 58px; left: 24px; max-width: 320px; }
  #legend-rows { display: flex; gap: 16px; margin-top: 8px; }
  #legend .row { display: flex; align-items: center; gap: 6px; }
  #legend .swatch { width: 5px; height: 18px; border-radius: 3px; flex: none; box-shadow: 0 0 12px currentColor; }
  #legend .row small { display: block; font-size: 9px; color: var(--muted); }
  #legend .cap { font-size: 10px; color: var(--muted); margin-top: 10px;
    border-top: 1px solid var(--border); padding-top: 9px; }
  #status { bottom: 78px; right: 24px; width: 210px; }
  #status .big { font-size: 23px; font-weight: 500; letter-spacing: -.035em; color: var(--bright); margin: 3px 0 8px; }
  #status .k { color: var(--muted); }
  #send-status { border-top: 1px solid var(--border); margin-top: 10px; padding-top: 10px; color: #bddee9; }
  #channel-status { color: var(--muted); font-size: 10px; margin-top: 5px; }
  #controls { position: absolute; z-index: 10; bottom: 23px; right: 24px; background: none; border: none;
    padding: 0; display: flex; gap: 7px; flex-wrap: wrap;
    justify-content: flex-end; max-width: 60%; }
  .btn { background: var(--panel); backdrop-filter: blur(9px);
    border: 1px solid var(--border); color: var(--bright); padding: 9px 13px;
    border-radius: 9px; font-size: 10px; cursor: pointer;
    transition: background 0.15s ease, border-color 0.15s ease; }
  .btn:hover, .btn:focus-visible { border-color: #86c7e9; background: #152c40; outline: none; }
  .btn.off { color: var(--muted); }
  #hint { position: absolute; z-index: 10; bottom: 27px; left: 28px; color: var(--muted); font-size: 10px; }
  #labels { position: absolute; inset: 0; z-index: 3; pointer-events: none; }
  .site-label { position: absolute; transform: translate(-50%,-100%); text-align: center; white-space: nowrap;
    color: var(--bright); padding: 5px 9px; border-radius: 8px; background: rgba(6,15,25,.68);
    border: 1px solid rgba(145,187,211,.12); box-shadow: 0 6px 18px rgba(0,0,0,.13); }
  .site-label strong { font-size: 12px; font-weight: 600; letter-spacing: .015em; }
  .site-label small { display: block; color: #8fa7bb; font-size: 9px; margin-top: 3px; }
  .site-label .site-dot { display: inline-block; height: 5px; width: 5px; border-radius: 50%; margin: 0 6px 2px 0;
    background: var(--site-color); opacity: .45; }
  .site-label.sending { border-color: color-mix(in srgb, var(--site-color) 35%, transparent); }
  .site-label.sending .site-dot { opacity: 1; box-shadow: 0 0 9px var(--site-color); }
  @media (max-width: 760px) {
    #masthead { top: 18px; left: 18px; } #masthead h1 { font-size: 20px; }
    #status { width: 155px; right: 12px; top: 12px; bottom: auto; padding: 10px; }
    #status-sub, #channel-status { display: none; }
    #legend { bottom: 65px; left: 12px; padding: 10px; } #legend .cap { display: none; }
    #hint { display: none; } #controls { right: 12px; bottom: 15px; max-width: 100%; }
  }
  #tip { position: absolute; z-index: 20; display: none; pointer-events: none;
    background: rgba(13, 17, 23, 0.96); border: 1px solid var(--border);
    border-radius: 8px; padding: 9px 11px; font-size: 0.70rem;
    line-height: 1.55; max-width: 270px; box-shadow: 0 10px 30px rgba(0,0,0,0.6); }
  #tip h4 { font-size: 0.82rem; color: var(--bright); margin-bottom: 3px; }
  #tip .k { color: var(--muted); }
  #tip table { width: 100%; border-collapse: collapse; margin-top: 5px; }
  #tip td { padding: 1px 0; }
  #tip td:last-child { text-align: right; color: var(--bright); }
</style>
<script src="__THREE_CDN__"></script>
<script src="__ORBIT_CDN__"></script>
</head>
<body>
<div id="stage"></div>
<div id="vignette"></div>
<div id="labels"></div>
<div id="masthead">
  <div class="eyebrow">Tokamak research network</div>
  <h1>Federation atlas</h1>
  <p>Four machines. Three channels. Shared learning.</p>
</div>

<div class="panel" id="legend">
  <div class="eyebrow">Policy streams</div>
  <div id="legend-rows"></div>
  <div class="cap">
    Stream density follows contribution weight.<br>
    White returns the aggregate. Broken links were refused.
  </div>
</div>

<div class="panel" id="status">
  <div class="eyebrow">Aggregate recipient</div>
  <div class="big" id="status-target">&mdash;</div>
  <div class="k" id="status-sub"></div><div id="send-status"></div><div id="channel-status"></div>
</div>

<div id="controls">
  <button class="btn" id="btn-flow">Animation: on</button>
  <button class="btn" id="btn-spin">Orbit: off</button>
  <button class="btn" id="btn-labels">Labels: on</button>
  <button class="btn" id="btn-reset">Reset view</button>
</div>

<div id="hint">Click machines to send &nbsp; / &nbsp; Drag to orbit &nbsp; / &nbsp; Scroll to explore</div>
<div id="tip"></div>

<script>
const DATA = __SCENE_DATA__;

// Channel decks. Both ends of a link sit at the same height, so a link is
// readable as "this channel" without following it.
const DECK = { thermal: 3.0, particle: 0.0, current: -3.0 };
const DEG = 0.45;
const BASE_Y = 5;
const sending = new Set(DATA.sending);
function project(lon, lat, y = 0) { return new THREE.Vector3(lon * DEG, y, -lat * DEG); }
const HUB_R     = 2.4;    // hub ring radius
const COIL_R    = 2.05;   // device coil radius (before per-device scale)

const stage = document.getElementById('stage');
const tip = document.getElementById('tip');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x060d17);


const camera = new THREE.PerspectiveCamera(
  48, stage.clientWidth / stage.clientHeight, 0.1, 500);
const HOME = new THREE.Vector3(-25, 45, 22);
const LOOK = new THREE.Vector3(-25, 0, -18);
camera.position.copy(HOME);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(stage.clientWidth, stage.clientHeight);
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
stage.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.autoRotate = false;
controls.autoRotateSpeed = 0.45;
controls.minDistance = 8;
controls.maxDistance = 240;
controls.target.copy(LOOK);
controls.maxPolarAngle = Math.PI * 0.48;
function resetCamera() {
  const fit=Math.max(1,1.75/(stage.clientWidth/stage.clientHeight));
  camera.position.copy(HOME).sub(LOOK).multiplyScalar(fit).add(LOOK);
  controls.target.copy(LOOK);
  controls.update();
}
resetCamera();

scene.add(new THREE.HemisphereLight(0xb7dcff, 0x101c2a, 1.15));
const key = new THREE.DirectionalLight(0xd5eaff, 1.6);
key.position.set(12, 22, 14);
scene.add(key);
const rim = new THREE.DirectionalLight(0x7bbdd1, 0.8);
rim.position.set(-16, -8, -12);
scene.add(rim);

const ocean = new THREE.Mesh(new THREE.PlaneGeometry(360 * DEG, 180 * DEG),
  new THREE.MeshBasicMaterial({color: new THREE.Color(0x091827).convertSRGBToLinear(), side: THREE.DoubleSide,toneMapped:false}));
ocean.rotation.x = -Math.PI / 2;
ocean.position.y = -0.04;
scene.add(ocean);
const graticule = [];
for (let lon = -180; lon <= 180; lon += 30) graticule.push(project(lon,-90),project(lon,90));
for (let lat = -90; lat <= 90; lat += 30) graticule.push(project(-180,lat),project(180,lat));
scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(graticule),
  new THREE.LineBasicMaterial({color: 0x264156,transparent:true,opacity:0.18})));
const LAND = __WORLD_LAND__;
LAND.forEach(function(ring) {
  const shape = new THREE.Shape(ring.map(p => new THREE.Vector2(p[0]*DEG,p[1]*DEG)));
  const land = new THREE.Mesh(new THREE.ShapeGeometry(shape),new THREE.MeshBasicMaterial({
    color:new THREE.Color(0x193344).convertSRGBToLinear(),side:THREE.DoubleSide,toneMapped:false}));
  land.rotation.x=-Math.PI/2;
  land.position.y=0.01;
  scene.add(land);
  const points = ring.map(p => project(p[0], p[1], 0.03));
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({color: 0x73949f,transparent:true,opacity:0.3})));
});

// ---------------------------------------------------------------- helpers

function hex(c) { return new THREE.Color(c).convertSRGBToLinear(); }

function labelSprite(text, color, size, weight) {
  // A bare family name, deliberately. The CSS font shorthand rejects an
  // unquoted multi-word family, and a rejected `ctx.font` silently leaves the
  // default 10px -- which draws readable-looking text into the corner of a
  // 64px canvas and scales it to a smear.
  const pad = 8, font = (weight || 600) + ' 42px sans-serif';
  const meas = document.createElement('canvas').getContext('2d');
  meas.font = font;
  const w = Math.ceil(meas.measureText(text).width) + pad * 2;
  const h = 64;
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d');
  ctx.font = font;
  ctx.textBaseline = 'middle';
  ctx.fillStyle = color;
  ctx.fillText(text, pad, h / 2);
  const tex = new THREE.CanvasTexture(cv);
  tex.minFilter = THREE.LinearFilter;
  // fog off: the scene is fogged for depth, and a label that fades out at the
  // far stations is a label that cannot be read when it matters.
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({
    map: tex, transparent: true, depthWrite: false, fog: false }));
  const s = size || 0.5;
  sp.scale.set((w / h) * s, s, 1);
  return sp;
}

const labelPool = [];
function addLabel(parent, text, color, size, pos, weight) {
  const sp = labelSprite(text, color, size, weight);
  sp.position.copy(pos);
  parent.add(sp);
  labelPool.push(sp);
  return sp;
}

// Screen-space names stay crisp; finer annotations appear on hover or zoom.
const screenLabels=[];
const labelPosition=new THREE.Vector3();
function siteLabel(parent,position,title,subtitle,color,deviceKey) {
  const el=document.createElement('div');
  el.className='site-label';
  el.style.setProperty('--site-color',color);
  const heading=document.createElement('strong');
  const dot=document.createElement('span'); dot.className='site-dot';
  heading.append(dot,document.createTextNode(title));
  const sub=document.createElement('small'); sub.textContent=subtitle;
  el.append(heading,sub);
  document.getElementById('labels').appendChild(el);
  screenLabels.push({parent,position,el,deviceKey});
}
function updateLabels() {
  screenLabels.forEach(label=>{
    labelPosition.copy(label.position);
    label.parent.localToWorld(labelPosition);
    labelPosition.project(camera);
    const visible=labelsOn && labelPosition.z>-1 && labelPosition.z<1 && Math.abs(labelPosition.x)<1 && Math.abs(labelPosition.y)<1;
    label.el.style.display=visible?'block':'none';
    if(visible) {
      label.el.style.left=(labelPosition.x+1)*stage.clientWidth/2+'px';
      label.el.style.top=(1-labelPosition.y)*stage.clientHeight/2+'px';
      label.el.classList.toggle('sending',label.deviceKey?sending.has(label.deviceKey):sending.size>0);
    }
  });
  labelPool.forEach(label=>{
    label.getWorldPosition(labelPosition);
    label.visible=labelsOn && (camera.position.distanceTo(labelPosition)<32 ||
      (hovered!==null && label.userData.device===hovered));
  });
}

// Exact digital-twin D profile (generate_3d_models.py), normalised by R.
function coilProfile(d, R) {
  const a = R * d.epsilon;
  const rCS = Math.max(0.14 / d.R * R, (R - a) * 0.45);
  const inner = Math.max(rCS * 1.15, (R - a) * 0.55);
  const outer = (R + a * 1.05) * 1.08;
  const top = a * d.kappa * 1.12;
  const v = (x,y) => new THREE.Vector3(x,y,0);
  const curve = new THREE.CurvePath();
  curve.add(new THREE.LineCurve3(v(inner,-top),v(inner,top)));
  curve.add(new THREE.QuadraticBezierCurve3(v(inner,top),v(R*0.9,top*1.06),v(outer*0.94,top*0.65)));
  curve.add(new THREE.QuadraticBezierCurve3(v(outer*0.94,top*0.65),v(outer,0),v(outer*0.94,-top*0.65)));
  curve.add(new THREE.QuadraticBezierCurve3(v(outer*0.94,-top*0.65),v(R*0.9,-top*1.06),v(inner,-top)));
  return {curve, outer, top, rCS, a};
}

function lineBetween(a, b, color, opacity) {
  const geo = new THREE.BufferGeometry().setFromPoints([a, b]);
  return new THREE.Line(geo, new THREE.LineBasicMaterial({
    color: hex(color), transparent: true, opacity: opacity }));
}

// ------------------------------------------------------------------- hub

const hub = new THREE.Group();
hub.position.copy(project(DATA.server.lon, DATA.server.lat, 10));
scene.add(hub);
scene.add(lineBetween(project(DATA.server.lon, DATA.server.lat),hub.position,'#52766c',0.6));

const hubRings = {};
DATA.clusters.forEach(function (c) {
  const y = DECK[c.key];
  const g = new THREE.Group();
  g.position.y = y;

  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(HUB_R, 0.085, 12, 120),
    new THREE.MeshStandardMaterial({
      color: hex(c.color), emissive: hex(c.color), emissiveIntensity: 0.75,
      metalness: 0.6, roughness: 0.28, transparent: true, opacity: 0.95 }));
  ring.rotation.x = Math.PI / 2;
  g.add(ring);

  const inner = new THREE.Mesh(
    new THREE.OctahedronGeometry(0.62, 0),
    new THREE.MeshStandardMaterial({
      color: hex(c.color), emissive: hex(c.color),
      emissiveIntensity: c.aggregated ? 0.95 : 0.15,
      metalness: 0.5, roughness: 0.4,
      transparent: true, opacity: c.aggregated ? 0.92 : 0.35,
      wireframe: !c.aggregated }));
  g.add(inner);
  g.userData.inner = inner;

  const disc = new THREE.Mesh(
    new THREE.RingGeometry(0.85, HUB_R - 0.12, 64),
    new THREE.MeshBasicMaterial({
      color: hex(c.color), transparent: true, opacity: 0.055,
      side: THREE.DoubleSide }));
  disc.rotation.x = -Math.PI / 2;
  g.add(disc);

  addLabel(g, c.label.toUpperCase() + ' CHANNEL', c.color, 0.46,
           new THREE.Vector3(0, 0.95, 0), 700);
  addLabel(g, c.payloadBytes + ' B payload', '#7d8590', 0.36,
           new THREE.Vector3(0, 0.52, 0), 500);


  hub.add(g);
  hubRings[c.key] = g;
});

// Short neutral stubs at each ring, NOT a column through all three. Thermal
// never aggregates with particle or current, so nothing in the scene may join
// the hubs -- a spine would draw exactly the coupling role matching forbids,
// and it would be the first thing a viewer read off the picture.
DATA.clusters.forEach(function (c) {
  const stub = new THREE.Mesh(
    new THREE.CylinderGeometry(0.035, 0.035, 1.1, 8),
    new THREE.MeshBasicMaterial({ color: 0x30363d, transparent: true, opacity: 0.5 }));
  stub.position.y = DECK[c.key];
  hub.add(stub);
});
siteLabel(hub,new THREE.Vector3(0,5.4,0),'Federation hub','Greenland · schematic site','#9cdded',null);

// --------------------------------------------------------------- devices

const deviceGroups = [];
const links = [];
const pickTargets = [];
const linkGroup = new THREE.Group();
scene.add(linkGroup);
let dynamicLabels = [];

DATA.devices.forEach(function(d) {
  const g = new THREE.Group();
  g.position.copy(project(d.lon + d.offsetLon, d.lat + d.offsetLat, BASE_Y));
  g.userData.device = d;
  scene.add(g);
  deviceGroups.push(g);
  const R = COIL_R * d.scale;
  const profile = coilProfile(d, R);
  const coilGeo = new THREE.TubeGeometry(profile.curve,64,Math.max(0.028 / d.R * R,profile.a*0.07),8,true);
  // Sixteen TF coils, as in the digital twin. Colors are a logical overlay.
  for(let k=0;k<16;k++) {
    const c = d.coils[k % d.coils.length];
    const mesh = new THREE.Mesh(coilGeo,new THREE.MeshStandardMaterial({
      color:hex(c.color).lerp(hex('#b0bdcd'),0.45),
      emissive:hex(c.color),emissiveIntensity:0.08,metalness:0.45,roughness:0.26}));
    mesh.rotation.y = k / 16 * Math.PI * 2;
    g.add(mesh);
  }
  const plasma = new THREE.Mesh(new THREE.TorusGeometry(R,profile.a,32,96),
    new THREE.MeshPhongMaterial({color:hex(d.color),emissive:hex(d.color),emissiveIntensity:0.35,
      shininess:90,transparent:true,opacity:0.2,depthWrite:false}));
  plasma.rotation.x = Math.PI/2;
  plasma.scale.z = d.kappa;
  g.add(plasma);
  g.userData.plasma = plasma;
  const solenoid = new THREE.Mesh(new THREE.CylinderGeometry(profile.rCS,profile.rCS,profile.top*2.1,24),
    new THREE.MeshStandardMaterial({color:hex('#586e81'),metalness:0.55,roughness:0.3}));
  g.add(solenoid);
  for(let winding=0;winding<7;winding++) {
    const band=new THREE.Mesh(new THREE.TorusGeometry(profile.rCS*1.015,0.025,6,36),
      new THREE.MeshStandardMaterial({color:0x98b5c5,metalness:0.55,roughness:0.3}));
    band.rotation.x=Math.PI/2; band.position.y=(winding/6-0.5)*profile.top*1.85; g.add(band);
  }
  d.coils.forEach(function(c, ci) {
    const outboard = profile.curve.getPoints(200).filter(p => p.x > R);
    c.agents.forEach(function(a,i) {
      const angle = (ci + i*3) / 16 * Math.PI*2;
      const p = outboard[Math.floor(outboard.length/2)].clone();
      p.applyAxisAngle(new THREE.Vector3(0,1,0),angle);
      const box = new THREE.Mesh(new THREE.BoxGeometry(0.4,0.4,0.4),
        new THREE.MeshStandardMaterial({color:a.available?hex(c.color):0x60666d,wireframe:!a.available}));
      box.position.copy(p); box.rotation.y = angle;
      g.add(box);
      g.add(lineBetween(p,new THREE.Vector3(),c.color,0.14));
      const agentLabel=addLabel(g,a.name+(a.available?'':' (off)'),c.color,0.36,p.clone().multiplyScalar(1.2).add(new THREE.Vector3(0,0.4,0)));
      agentLabel.userData.device=d.key;
    });
  });
  const pin = project(d.lon,d.lat,0.1);
  const marker = new THREE.Mesh(new THREE.SphereGeometry(0.22,12,8),new THREE.MeshBasicMaterial({color:hex(d.color)}));
  marker.position.copy(pin); scene.add(marker);
  scene.add(lineBetween(pin,g.position,d.color,0.32));
  const footprint=new THREE.Mesh(new THREE.RingGeometry(0.42,0.5,48),
    new THREE.MeshBasicMaterial({color:hex(d.color),transparent:true,opacity:0.5,side:THREE.DoubleSide}));
  footprint.rotation.x=-Math.PI/2; footprint.position.copy(pin); scene.add(footprint);
  siteLabel(g,new THREE.Vector3(0,profile.top+1.4,0),d.label,
    d.place+(d.isTarget?' · recipient':''),d.color,d.key);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(profile.outer*1.15,0.035,8,96),
    new THREE.MeshBasicMaterial({color:hex(d.color),transparent:true,opacity:0.2}));
  ring.rotation.x = Math.PI/2; ring.position.y = -profile.top-0.4; g.add(ring);
  g.userData.sendRing = ring;
  const plinth=new THREE.Mesh(new THREE.CylinderGeometry(profile.outer*1.1,profile.outer*1.13,0.12,64),
    new THREE.MeshStandardMaterial({color:hex('#102638'),metalness:0.35,roughness:0.55,transparent:true,opacity:0.8}));
  plinth.position.y=ring.position.y-0.1; g.add(plinth);
  const pick = new THREE.Mesh(new THREE.CylinderGeometry(profile.outer*1.1,profile.outer*1.1,profile.top*2.4,16),
    new THREE.MeshBasicMaterial({transparent:true,opacity:0,depthWrite:false,side:THREE.DoubleSide}));
  pick.userData.device=d; g.add(pick); pickTargets.push(pick);
});

// Soft additive sprites give each moving bunch a bright core and a fading halo.
const particleCanvas = document.createElement('canvas');
particleCanvas.width = particleCanvas.height = 64;
const particleContext = particleCanvas.getContext('2d');
const particleGradient = particleContext.createRadialGradient(32,32,0,32,32,32);
particleGradient.addColorStop(0,'rgba(255,255,255,1)');
particleGradient.addColorStop(0.16,'rgba(255,255,255,0.95)');
particleGradient.addColorStop(0.4,'rgba(255,255,255,0.25)');
particleGradient.addColorStop(1,'rgba(255,255,255,0)');
particleContext.fillStyle=particleGradient;
particleContext.fillRect(0,0,64,64);
const particleTexture=new THREE.CanvasTexture(particleCanvas);
const streamPoint=new THREE.Vector3();
const streamTangent=new THREE.Vector3();
const streamSide=new THREE.Vector3();
const streamUp=new THREE.Vector3();
const vertical=new THREE.Vector3(0,1,0);

function updateStream(stream,dt) {
  stream.phase=(stream.phase+stream.speed*dt)%1;
  const positions=stream.cloud.geometry.attributes.position;
  for(let bunch=0;bunch<stream.bunches;bunch++) {
    for(let tail=0;tail<stream.tailSize;tail++) {
      const travel=(stream.phase+bunch/stream.bunches-tail*0.0017+1)%1;
      const u=stream.dir>0?travel:1-travel;
      stream.curve.getPoint(u,streamPoint);
      stream.curve.getTangent(u,streamTangent);
      streamSide.crossVectors(streamTangent,vertical).normalize();
      streamUp.crossVectors(streamSide,streamTangent).normalize();
      const twist=tail*0.65+bunch*2.4+stream.phase*50;
      const spread=0.018+0.075*tail/stream.tailSize;
      streamPoint.addScaledVector(streamSide,Math.cos(twist)*spread);
      streamPoint.addScaledVector(streamUp,Math.sin(twist)*spread);
      positions.setXYZ(bunch*stream.tailSize+tail,streamPoint.x,streamPoint.y,streamPoint.z);
    }
  }
  positions.needsUpdate=true;
}

function rebuildLinks() {
  // Dispose old GPU resources on every click; the stationary scene stays intact.
  linkGroup.traverse(o => {
    if(o.geometry) o.geometry.dispose();
    if(o.material) { if(o.material.map && o.material.map!==particleTexture) o.material.map.dispose(); o.material.dispose(); }
  });
  linkGroup.clear(); links.length=0;
  dynamicLabels.forEach(l => {const i=labelPool.indexOf(l); if(i>=0) labelPool.splice(i,1);});
  dynamicLabels=[];
  const mask=DATA.deviceOrder.map(k => sending.has(k)?'1':'0').join('');
  const state=DATA.participation[mask];
  if(!state) throw new Error('Missing participation state '+mask);
  DATA.clusters.forEach(c => {
    const active=state.clusters[c.key].aggregated;
    const mat=hubRings[c.key].userData.inner.material;
    mat.wireframe=!active; mat.opacity=active?0.92:0.25;
    mat.emissiveIntensity=active?0.95:0.15;
  });
  function packets(curve,color,weight,dir) {
    const bunches=dir<0?9:Math.max(5,Math.round(7+22*weight));
    const tailSize=26, count=bunches*tailSize;
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(count*3),3).setUsage(THREE.DynamicDrawUsage));
    const colors=new Float32Array(count*3);
    const tint=hex(color);
    for(let i=0;i<count;i++) {
      const tail=i%tailSize;
      const fade=Math.pow(1-tail/tailSize,1.7);
      const head=tail<3?0.6:0;
      colors[i*3]=(tint.r*(1-head)+head)*fade;
      colors[i*3+1]=(tint.g*(1-head)+head)*fade;
      colors[i*3+2]=(tint.b*(1-head)+head)*fade;
    }
    geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));
    const cloud=new THREE.Points(geometry,new THREE.PointsMaterial({
      map:particleTexture,size:dir<0?0.48:0.58,vertexColors:true,
      transparent:true,blending:THREE.AdditiveBlending,depthWrite:false,opacity:0.95}));
    cloud.frustumCulled=false;
    cloud.visible=flowOn;
    linkGroup.add(cloud);
    const stream={curve,cloud,bunches,tailSize,phase:dir<0?0.5:0,speed:0.09+0.18*weight,dir};
    updateStream(stream,0);
    links.push(stream);
  }
  deviceGroups.forEach(g => {
    const d=g.userData.device;
    g.userData.sendRing.material.opacity=sending.has(d.key)?0.8:0.18;
    d.coils.forEach(c => {
      Object.assign(c,state.devices[d.key][c.cluster]);
      const start=g.position.clone(); start.y += DECK[c.cluster]*0.35;
      const end=hub.position.clone(); end.y += DECK[c.cluster];
      const mid=start.clone().lerp(end,0.5); mid.y+=6;
      const curve=new THREE.QuadraticBezierCurve3(start,mid,end);
      if(!c.excluded) {
        const w=c.weight;
        if(c.admissible) {
          // A faint beam envelope replaces the solid cable; moving particles dominate.
          linkGroup.add(new THREE.Mesh(new THREE.TubeGeometry(curve,64,0.06+0.07*w,6,false),
            new THREE.MeshBasicMaterial({color:hex(c.color),transparent:true,opacity:0.045+0.05*w,
              blending:THREE.AdditiveBlending,depthWrite:false})));
          if(w>0) packets(curve,c.color,w,1);
        } else {
          const pts=curve.getPoints(48), segments=[];
          for(let i=0;i<48;i+=2) if(i<20||i>28) segments.push(pts[i],pts[i+1]);
          linkGroup.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(segments),
            new THREE.LineBasicMaterial({color:0x7d8590})));
        }
        const label=addLabel(linkGroup,c.admissible?(w*100).toFixed(1)+'%':'refused',
          c.admissible?c.color:'#f85149',0.42,curve.getPoint(0.4));
        label.userData.device=d.key;
        label.visible=labelsOn; dynamicLabels.push(label);
      }
      // The target may receive even when it is silent or its own update is refused.
      if(d.isTarget && state.clusters[c.cluster].aggregated) packets(curve,'#ffffff',0.8,-1);
    });
  });
  document.getElementById('send-status').textContent=sending.size+' / '+DATA.devices.length+' machines sending';
  document.getElementById('channel-status').innerHTML=DATA.clusters.map(c => {
    const diag=state.clusters[c.key];
    return c.label+': '+(diag.aggregated?diag.effectivePeers.toFixed(2)+' effective peers':'no aggregate');
  }).join('<br>');
  if(hovered) renderTip(DATA.devices.find(d=>d.key===hovered));
}

// --------------------------------------------------------------- overlays

(function buildLegend() {
  const host = document.getElementById('legend-rows');
  DATA.clusters.forEach(function (c) {
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = '<span class="swatch" style="background:' + c.color + ';color:' + c.color +
      '"></span><span>' + c.label + '<small>' + c.payloadBytes + ' B / update</small></span>';
    host.appendChild(row);
  });
})();

(function buildStatus() {
  const d = DATA.devices.filter(function (x) { return x.isTarget; })[0];
  document.getElementById('status-target').textContent = d ? d.label : DATA.target;
  document.getElementById('status-sub').innerHTML =
    'kernel bandwidth <b>' + DATA.bandwidth.toFixed(3) + '</b><br>' +
    'rule <b>' + DATA.rule + '</b> &middot; round <b>' + DATA.round + '</b><br>' +
    (DATA.useSimilarity ? 'similarity-weighted' : 'uniform (baseline 2)');
})();

let flowOn = true, labelsOn = true;
const btnFlow = document.getElementById('btn-flow');
const btnSpin = document.getElementById('btn-spin');
const btnLabels = document.getElementById('btn-labels');

btnFlow.onclick = function () {
  flowOn = !flowOn;
  btnFlow.textContent = 'Animation: ' + (flowOn ? 'on' : 'off');
  btnFlow.classList.toggle('off', !flowOn);
  links.forEach(function (l) {
    l.cloud.visible = flowOn;
  });
};
btnSpin.onclick = function () {
  controls.autoRotate = !controls.autoRotate;
  btnSpin.textContent = 'Orbit: ' + (controls.autoRotate ? 'on' : 'off');
  btnSpin.classList.toggle('off', !controls.autoRotate);
};
btnSpin.classList.add('off');
btnLabels.onclick = function () {
  labelsOn = !labelsOn;
  btnLabels.textContent = 'Labels: ' + (labelsOn ? 'on' : 'off');
  btnLabels.classList.toggle('off', !labelsOn);
  labelPool.forEach(function (s) { s.visible = labelsOn; });
};
document.getElementById('btn-reset').onclick = function () {
  resetCamera();
};

// ------------------------------------------------------------------ hover

const ray = new THREE.Raycaster();
const ptr = new THREE.Vector2();
let hovered = null;
let press = null;
const activePointers = new Set();
renderer.domElement.addEventListener('pointerdown', function(e) {
  activePointers.add(e.pointerId);
  press=activePointers.size===1 && e.button===0?{id:e.pointerId,x:e.clientX,y:e.clientY,time:performance.now(),moved:false}:null;
});
renderer.domElement.addEventListener('pointermove', function(e) {
  if(press && Math.hypot(e.clientX-press.x,e.clientY-press.y)>5) press.moved=true;
});
renderer.domElement.addEventListener('pointercancel', function(e) {activePointers.delete(e.pointerId);press=null;});
renderer.domElement.addEventListener('pointerup', function(e) {
  activePointers.delete(e.pointerId);
  const p=press; press=null;
  if(!p || p.id!==e.pointerId || p.moved || performance.now()-p.time>600) return;
  const r=renderer.domElement.getBoundingClientRect();
  ptr.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);
  ray.setFromCamera(ptr,camera);
  const hits=ray.intersectObjects(pickTargets,false);
  if(!hits.length) return;
  const key=hits[0].object.userData.device.key;
  if(sending.has(key)) sending.delete(key); else sending.add(key);
  rebuildLinks();
});

renderer.domElement.addEventListener('pointermove', function (e) {
  const r = renderer.domElement.getBoundingClientRect();
  ptr.x = ((e.clientX - r.left) / r.width) * 2 - 1;
  ptr.y = -((e.clientY - r.top) / r.height) * 2 + 1;
  ray.setFromCamera(ptr, camera);
  const hit = ray.intersectObjects(pickTargets, false);
  if (hit.length) {
    const d = hit[0].object.userData.device;
    if (hovered !== d.key) { hovered = d.key; renderTip(d); }
    tip.style.display = 'block';
    tip.style.left = Math.min(e.clientX + 16, window.innerWidth - 290) + 'px';
    tip.style.top = Math.min(e.clientY + 14, window.innerHeight - 220) + 'px';
  } else {
    hovered = null;
    tip.style.display = 'none';
  }
});
renderer.domElement.addEventListener('pointerleave', function () {
  hovered = null; tip.style.display = 'none';
});

function renderTip(d) {
  let rows = '';
  rows += '<tr><td class="k">R / a / B0</td><td>' + d.R.toFixed(2) + ' m &middot; ' +
          d.a.toFixed(2) + ' m &middot; ' + d.B0.toFixed(1) + ' T</td></tr>';
  rows += '<tr><td class="k">Ip</td><td>' + d.Ip.toFixed(2) + ' MA</td></tr>';
  rows += '<tr><td class="k">rho*</td><td>' + d.state.rho_star.toExponential(2) + '</td></tr>';
  rows += '<tr><td class="k">nu*</td><td>' + d.state.nu_star.toFixed(3) + '</td></tr>';
  rows += '<tr><td class="k">beta_N</td><td>' + d.state.beta_N.toFixed(2) + '</td></tr>';
  rows += '<tr><td class="k">q95</td><td>' + d.state.q95.toFixed(2) + '</td></tr>';
  rows += '<tr><td class="k">distance to target</td><td>' + d.distance.toFixed(3) + '</td></tr>';
  if (d.roundAge) {
    rows += '<tr><td class="k">update age</td><td>' + d.roundAge + ' rounds</td></tr>';
  }
  if (d.violationRate > 0) {
    rows += '<tr><td class="k">violation rate</td><td>' +
            (d.violationRate * 100).toFixed(0) + '%</td></tr>';
  }
  d.coils.forEach(function (c) {
    let cell;
    if (c.admissible) {
      cell = (c.weight * 100).toFixed(1) + '%';
    } else if (c.excluded) {
      cell = '<span class="k">not buffered</span>';
    } else {
      cell = '<span style="color:#f85149">refused by gate</span>';
    }
    rows += '<tr><td class="k" style="color:' + c.color + '">' + c.label +
            '</td><td>' + cell + '</td></tr>';
  });
  tip.innerHTML = '<h4 style="color:' + d.color + '">' + d.label + '</h4>' +
                  '<div class="k">' + d.place + ' · ' + d.lat.toFixed(2) + ', ' + d.lon.toFixed(2) + '<br>' + (sending.has(d.key)?'Sending':'Silent') + '</div><table>' + rows + '</table>';
}

// ----------------------------------------------------------------- render

const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  const t = clock.elapsedTime;

  if (flowOn) {
    links.forEach(function (l) { updateStream(l,dt); });
  }

  deviceGroups.forEach(function (g, i) {
    // Breathing plasma, matching the per-device viewports' shader idea.
    const pulse = Math.sin(t * 1.4 + i) * 0.13 + 0.87;
    g.userData.plasma.material.opacity = 0.12 * pulse + 0.12;

    if (g.userData.sel) { g.userData.sel.rotation.z += dt * 0.6; }
  });

  Object.keys(hubRings).forEach(function (k, i) {
    const inner = hubRings[k].userData.inner;
    inner.rotation.y += dt * (0.4 + 0.12 * i);
    inner.rotation.x += dt * 0.22;
  });

  controls.update();
  scene.updateMatrixWorld(true);
  updateLabels();
  renderer.render(scene, camera);
}
rebuildLinks();
animate();

window.addEventListener('resize', function () {
  camera.aspect = stage.clientWidth / stage.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(stage.clientWidth, stage.clientHeight);
});
</script>
</body>
</html>
"""


def build_scene_html(payload: dict) -> str:
    """Render the scene for one federation graph.

    ``json.dumps`` rather than an f-string: the template is mostly JavaScript
    braces, and formatting it would mean escaping every one of them.
    """
    return (
        _TEMPLATE.replace("__THREE_CDN__", _THREE_CDN)
        .replace("__ORBIT_CDN__", _ORBIT_CDN)
        .replace("__SCENE_DATA__", json.dumps(payload, allow_nan=False))
        .replace("__WORLD_LAND__", json.dumps(LAND))
    )
