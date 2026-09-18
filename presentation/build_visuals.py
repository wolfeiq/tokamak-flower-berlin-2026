"""Export existing 3D views for the presentation; never run TORAX or Flower."""

import sys
from shutil import copyfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz.federation_data import build_graph, participation_sweep, scene_payload
from viz.federation_scene import build_scene_html


def main():
    output = ROOT / "presentation" / "visuals"
    output.mkdir(exist_ok=True)
    copyfile(ROOT / "results" / "audit" / "system_design.png", output / "marl-architecture.png")
    graph = build_graph(participants=())
    payload = scene_payload(graph, participation=participation_sweep(graph))
    html = build_scene_html(payload)
    html = html.replace(
        "Four machines. Three channels. Shared learning.",
        "Synthetic policy updates · nominal operating points · no live connections",
    )
    html = html.replace("</style>", """
    /* Keep the map visible when embedded in a narrow presentation pane. */
    @media (max-width: 760px) {
      #masthead p { max-width: 300px; font-size: 9px; }
      #status { top: auto; bottom: 64px; width: 145px; }
      #legend { max-width: 240px; }
      #legend-rows { gap: 10px; }
    }
    </style>""")
    (output / "atlas.html").write_text(html, encoding="utf-8")
    assembly = (ROOT / "assets" / "3d" / "html" / "diiid_like.html").read_text(
        encoding="utf-8"
    )
    # Fit the existing geometry in a portrait iframe as well as a wide screen.
    assembly = assembly.replace(
        "camera.position.set(200, 250, 520);",
        "camera.position.set(200, 250, 520).multiplyScalar("
        "1.9 / Math.min(1, window.innerWidth / window.innerHeight));",
    )
    assembly = assembly.replace("</style>", """
    .legend-overlay { top: 12px; bottom: auto; right: 12px; font-size: 9px; flex-wrap: wrap; gap: 6px; }
    .controls-overlay { max-width: 100%; }
    .hero .legend-overlay, .hero .controls-overlay { display: none; }
    </style>""")
    assembly = assembly.replace("</head>", """
    <script>if (location.hash === '#hero') document.documentElement.classList.add('hero');</script>
    </head>""")
    assembly = assembly.replace("    animate();", """
    if (location.hash === '#hero') {
      scene.background.setHex(0x08090b);
      gridHelper.visible = false;
      keyLight.color.setHex(0xacc8ff);
      fillLight.color.setHex(0xff9e74);
      plasmaMaterial.color.setHex(0xff7755);
      plasmaMaterial.emissive.setHex(0xff7755);
      plasmaLight.color.setHex(0xff7755);
      controls.enablePan = false;
      controls.autoRotate = !matchMedia('(prefers-reduced-motion: reduce)').matches;
      controls.autoRotateSpeed = 0.24;
      function frameHero() {
        const w = innerWidth, h = innerHeight, narrow = w <= 850;
        camera.position.set(200, 190, 520).multiplyScalar(
          (narrow ? 1.30 : 1.14) / Math.min(1, w / h));
        camera.setViewOffset(w, h, narrow ? -w * 0.06 : -w * 0.23,
          narrow ? -h * 0.17 : 0, w, h);
        camera.updateProjectionMatrix();
      }
      frameHero();
      addEventListener('resize', frameHero);
    }
    animate();""")
    (output / "diiid_like.html").write_text(assembly, encoding="utf-8")
    print("Exported federation atlas and DIII-D-like assembly to presentation/visuals")


if __name__ == "__main__":
    main()
