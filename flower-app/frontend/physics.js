'use strict';
/* In-browser port of fusion_agent/thermal/profiles.py — the reduced 1-D
 * steady-state transport solve, so the audience can run the physics inside the
 * investigation UI. Same constants, same bisection; a REDUCED model, not TORAX,
 * and labelled as such wherever its output appears. */
(() => {
const CRITICAL_R_OVER_LT = 4.0;
const STIFFNESS = 1.5;
const CHI_FLOOR_FRACTION = 0.2;
const E_CHARGE = 1.602176634e-19;
const M_P = 1.67262192e-27;
const RHO_S_COEFF = Math.sqrt(M_P / E_CHARGE);
const KEV_TO_J = 1e3 * E_CHARGE;

function gyroBohmChi(T_keV, B0, a, A_i = 2.5) {
  const T_eV = Math.max(T_keV * 1e3, 1.0);
  const rho_s = RHO_S_COEFF * Math.sqrt(A_i * T_eV) / B0;
  const c_s = Math.sqrt(E_CHARGE * T_eV / (A_i * M_P));
  return rho_s * rho_s * c_s / a;
}
const chiOfGradient = (chiGB, rOverLt) =>
  chiGB * (CHI_FLOOR_FRACTION + STIFFNESS * Math.pow(Math.max(0, rOverLt - CRITICAL_R_OVER_LT), 1.5));

function solve(dev, powerFraction) {
  const n = dev.n_rho || 101, a = dev.a_minor, R = dev.R_major;
  const r = Array.from({length: n}, (_, i) => (dev.rho_boundary || 0.85) * a * i / (n - 1));
  const n_e = r.map(x => dev.n_e * (1 - 0.8 * (x / a) ** 2));
  const width = 0.35 * a;
  const shape = r.map(x => Math.exp(-((x / width) ** 2)));
  // Cylindrical dV = 4 pi^2 R kappa r dr; trapezoid, as in profiles.py.
  let integral = 0;
  for (let i = 1; i < n; i++)
    integral += (r[i] - r[i-1]) * 0.5 * (shape[i]*r[i] + shape[i-1]*r[i-1]) * 4 * Math.PI**2 * R * dev.elongation;
  const S = shape.map(v => v * dev.P_aux * powerFraction / Math.max(integral, 1e-30));

  const q = new Array(n).fill(0);
  let acc = 0;
  for (let i = 1; i < n; i++) {
    acc += (r[i] - r[i-1]) * 0.5 * (S[i]*r[i] + S[i-1]*r[i-1]);
    q[i] = acc / r[i];
  }
  const chiGB = gyroBohmChi(dev.T_e_keV, dev.B_0, a);
  const T = new Array(n).fill(0), chi = new Array(n).fill(0);
  T[n-1] = dev.T_pedestal_keV * KEV_TO_J;
  for (let i = n - 2; i >= 0; i--) {
    const step = r[i+1] - r[i], qm = 0.5*(q[i]+q[i+1]), nm = 0.5*(n_e[i]+n_e[i+1]);
    const floor = Math.max(chiGB * CHI_FLOOR_FRACTION, 1e-12);
    let lo = T[i+1], hi = T[i+1] + step * qm / (nm * floor), c = floor;
    if (qm <= 0 || hi <= lo) { T[i] = T[i+1]; chi[i] = floor; continue; }
    for (let k = 0; k < 200; k++) {                    // bisection: flux is monotone in T_in
      const mid = 0.5*(lo+hi), grad = (mid - T[i+1])/step, Tm = Math.max(0.5*(mid+T[i+1]), 1e-300);
      c = Math.max(chiOfGradient(chiGB, R*Math.abs(grad)/Tm), 1e-12);
      if (nm * c * grad < qm) lo = mid; else hi = mid;
      if (hi - lo <= 1e-14 * Math.abs(hi)) break;
    }
    T[i] = 0.5*(lo+hi); chi[i] = c;
  }
  chi[n-1] = chi[n-2];
  const med = arr => { const s=[...arr].slice(5, n-5).sort((x,y)=>x-y); return s[s.length>>1]; };
  return { r, rho: r.map(x=>x/a), T_keV: T.map(v=>v/KEV_TO_J), chi, q, n_e,
           chiRef: med(chi), T_axis: T[0]/KEV_TO_J,
           rOverLtMid: R*Math.abs((T[45]-T[55])/(r[55]-r[45]))/Math.max(T[50],1e-300) };
}

const svgNS = 'http://www.w3.org/2000/svg';
function drawProfiles(el, runs) {
  el.textContent = '';
  const W = 560, H = 230, PL = 42, PB = 26, PT = 12, PR = 12;
  const maxT = Math.max(...runs.map(s => s.T_axis), 0.1);
  const line = (pts, cls, dash) => {
    const p = document.createElementNS(svgNS, 'polyline');
    p.setAttribute('points', pts.map(([x,y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' '));
    p.setAttribute('class', cls); if (dash) p.setAttribute('stroke-dasharray', dash);
    el.appendChild(p);
  };
  const axis = document.createElementNS(svgNS, 'path');
  axis.setAttribute('d', `M${PL} ${PT} V${H-PB} H${W-PR}`);
  axis.setAttribute('class', 'axis'); el.appendChild(axis);
  runs.forEach((s, k) => line(
    s.rho.map((x, i) => [PL + x*(W-PL-PR)/0.85, H-PB - (s.T_keV[i]/maxT)*(H-PB-PT)]),
    'trace trace-' + k));
  const lab = (x, y, text, anchor) => {
    const t = document.createElementNS(svgNS, 'text');
    t.setAttribute('x', x); t.setAttribute('y', y); if (anchor) t.setAttribute('text-anchor', anchor);
    t.setAttribute('class', 'axis-label'); t.textContent = text; el.appendChild(t);
  };
  lab(PL, H-8, '0'); lab(W-PR, H-8, 'r / a = 0.85', 'end');
  lab(PL-6, PT+8, maxT.toFixed(1) + ' keV', 'end'); lab(PL-6, H-PB, '0', 'end');
}

function fmtChi(v) { return v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2); }

async function init() {
  const root = document.getElementById('physics');
  if (!root) return;
  let devices;
  try { devices = await (await fetch('/api/devices')).json(); }
  catch { root.querySelector('.panel').textContent = 'Device parameters unavailable.'; return; }
  const $ = id => document.getElementById(id);
  const state = { site: 'A', power: 1.0, assumed: 1.6 };
  function render() {
    const dev = devices[state.site];
    const run = solve(dev, state.power);
    const nominal = solve(dev, 1.0);
    drawProfiles($('physics-chart'), [run, nominal]);
    const apparent = run.chiRef * state.assumed;   // q scales linearly with S
    const ratio = apparent / nominal.chiRef;
    const usable = run.rOverLtMid >= CRITICAL_R_OVER_LT * 0.75;
    $('physics-taxis').textContent = run.T_axis.toFixed(2) + ' keV';
    $('physics-chi').textContent = fmtChi(run.chiRef) + ' m²/s';
    $('physics-apparent').textContent = usable ? fmtChi(apparent) + ' m²/s' : 'unidentifiable';
    $('physics-verdict').textContent = !usable
      ? 'Profile below the critical gradient: transport cannot be identified from it. This is facility C’s situation.'
      : state.assumed > 1.25
        ? `Assuming commanded power was delivered inflates χ by ×${state.assumed.toFixed(2)} — an apparent transport anomaly (ratio ${ratio.toFixed(2)}) with nothing wrong with the transport. This is facility A’s situation.`
        : 'With the delivered power verified, the recovered χ sits near this device’s own reference — facility B after its source audit.';
    $('physics-device').textContent = `${dev.name} · B₀ ${dev.B_0} T · a ${dev.a_minor} m · R ${dev.R_major} m`;
    const twin = $('twin-frame');
    if (twin && twin.dataset.device !== dev.name) {
      twin.dataset.device = dev.name;
      if (twin.dataset.on === '1') twin.src = '/twin/' + dev.name + '.html';
    }
  }
  $('physics-site').addEventListener('change', e => { state.site = e.target.value; render(); });
  $('physics-power').addEventListener('input', e => {
    state.power = Math.pow(10, parseFloat(e.target.value));
    $('physics-power-value').textContent = state.power >= 0.1 ? state.power.toFixed(2) + '×' : (state.power*100).toFixed(1) + '%';
    render();
  });
  $('physics-assumed').addEventListener('input', e => {
    state.assumed = parseFloat(e.target.value);
    $('physics-assumed-value').textContent = '×' + state.assumed.toFixed(2);
    render();
  });
  const twinButton = $('twin-load');
  if (twinButton) twinButton.addEventListener('click', () => {
    const twin = $('twin-frame');
    twin.dataset.on = '1';
    twin.src = '/twin/' + devices[state.site].name + '.html';
    twin.hidden = false; twinButton.hidden = true;
  });
  render();
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
