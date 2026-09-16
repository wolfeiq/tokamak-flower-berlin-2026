"""Export existing 3D views for the presentation; never run TORAX or Flower."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz.federation_data import build_graph, participation_sweep, scene_payload
from viz.federation_scene import build_scene_html


def main():
    output = ROOT / "presentation" / "visuals"
    output.mkdir(exist_ok=True)
    graph = build_graph(participants=())
    payload = scene_payload(graph, participation=participation_sweep(graph))
    html = build_scene_html(payload)
    html = html.replace(
        "Four machines. Three channels. Shared learning.",
        "Illustrative policy exchange · nominal operating points · no live connections",
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
    </style>""")
    (output / "diiid_like.html").write_text(assembly, encoding="utf-8")
    print("Exported federation atlas and DIII-D-like assembly to presentation/visuals")


if __name__ == "__main__":
    main()
