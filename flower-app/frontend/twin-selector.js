'use strict';
// Compact, selectable view of the existing registry-generated reactor assembly.
// The full assembly viewer keeps its original controls and camera.
if (location.hash === '#selector') {
  document.querySelectorAll('.controls-overlay, .legend-overlay').forEach(el => el.style.display = 'none');
  const send = type => parent.postMessage({type}, location.origin);
  try {
    const bounds = new THREE.Box3().setFromObject(tokamakAssembly);
    const sphere = bounds.getBoundingSphere(new THREE.Sphere());
    function fitDevice() {
      const halfFov = Math.atan(Math.tan(camera.fov * Math.PI / 360) * Math.min(1, camera.aspect));
      const distance = sphere.radius / Math.sin(halfFov) * 1.12;
      controls.target.copy(sphere.center);
      camera.position.copy(sphere.center).add(new THREE.Vector3(1.1, 0.65, 1.5).normalize().multiplyScalar(distance));
      controls.update();
    }
    scene.background.set(0x0c0f14);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    controls.enableZoom = false;
    controls.enablePan = false;
    fitDevice();
    window.addEventListener('resize', fitDevice);
    let down = null;
    renderer.domElement.style.cursor = 'grab';
    renderer.domElement.addEventListener('pointerdown', e => { down = [e.clientX,e.clientY]; });
    renderer.domElement.addEventListener('pointerup', e => {
      if (down && Math.hypot(e.clientX-down[0], e.clientY-down[1]) < 7) send('fusion:select-facility');
      down = null;
    });
    renderer.domElement.addEventListener('pointercancel', () => { down = null; });
  } catch {
    const status = document.getElementById('loading');
    if (status) {
      status.style.display = 'block';
      status.textContent = '3D unavailable. Select the facility below.';
    }
  }
}
