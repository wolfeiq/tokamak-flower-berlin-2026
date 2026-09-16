"""Convert STL models to valid ISO-10303-21 STEP (AP203 Faceted B-Rep) files."""

from __future__ import annotations

import struct
from pathlib import Path


def stl_to_step(stl_path: Path, step_path: Path, model_name: str = "tokamak_plasma") -> None:
    data = stl_path.read_bytes()
    num_triangles = struct.unpack("<I", data[80:84])[0]
    print(f"Converting {stl_path.name} ({num_triangles} triangles) to STEP...")

    header_lines = [
        "ISO-10303-21;",
        "HEADER;",
        f"FILE_DESCRIPTION(('{model_name} CAD Model','STEP AP203'),'2;1');",
        f"FILE_NAME('{step_path.name}','2026-09-16T12:00:00',('Antigravity'),('Tokamak Federation'),'Tokamak STEP Generator','CAD','');",
        "FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));",
        "ENDSEC;",
        "DATA;",
        "#1=APPLICATION_CONTEXT('configuration controlled 3D design of mechanical parts and assemblies');",
        "#2=APPLICATION_PROTOCOL_DEFINITION('international standard','config_control_design',1994,#1);",
        "#3=PRODUCT_CONTEXT('',#1,'mechanical');",
        f"#4=PRODUCT('{model_name}','{model_name}','',(#3));",
        "#5=PRODUCT_DEFINITION_FORMATION('','',#4);",
        "#6=PRODUCT_DEFINITION('design','',#5,#3);",
        "#7=PRODUCT_DEFINITION_SHAPE('','',#6);",
        "#8=DIRECTION('',(0.,0.,1.));",
        "#9=DIRECTION('',(1.,0.,0.));",
        "#10=CARTESIAN_POINT('',(0.,0.,0.));",
        "#11=AXIS2_PLACEMENT_3D('',#10,#8,#9);",
        "#12=(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#16)) GLOBAL_UNIT_ASSIGNED_CONTEXT((#13,#14,#15)) REPRESENTATION_CONTEXT('tokamak','3D'));",
        "#13=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));",
        "#14=(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.));",
        "#15=(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT());",
        "#16=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-05),#13,'closure',#12);",
    ]

    eid = 20
    point_map = {}
    cartesian_points = []
    lines = []
    faces = []

    offset = 84
    for _ in range(num_triangles):
        vals = struct.unpack("<12f", data[offset: offset + 48])
        offset += 50

        # Scale cm to mm (* 10)
        tri_pt_ids = []
        for v in range(3):
            pt = (
                round(vals[3 + v * 3] * 10.0, 3),
                round(vals[4 + v * 3] * 10.0, 3),
                round(vals[5 + v * 3] * 10.0, 3),
            )
            if pt not in point_map:
                point_map[pt] = eid
                cartesian_points.append(
                    f"#{eid}=CARTESIAN_POINT('',({pt[0]:.3f},{pt[1]:.3f},{pt[2]:.3f}));"
                )
                eid += 1
            tri_pt_ids.append(point_map[pt])

        p1, p2, p3 = tri_pt_ids
        poly_id = eid
        lines.append(f"#{poly_id}=POLY_LOOP('',(#{p1},#{p2},#{p3}));")
        eid += 1

        bound_id = eid
        lines.append(f"#{bound_id}=FACE_OUTER_BOUND('',#{poly_id},.T.);")
        eid += 1

        face_id = eid
        lines.append(f"#{face_id}=FACETED_PRIMITIVE_FACE('',(#{bound_id}));")
        faces.append(f"#{face_id}")
        eid += 1

    total_lines = header_lines + cartesian_points + lines

    shell_id = eid
    eid += 1
    face_list = ",".join(faces)
    total_lines.append(f"#{shell_id}=CLOSED_SHELL('',({face_list}));")

    brep_id = eid
    eid += 1
    total_lines.append(f"#{brep_id}=FACETED_BREP('{model_name}',#{shell_id});")

    shape_rep_id = eid
    eid += 1
    total_lines.append(
        f"#{shape_rep_id}=SHAPE_REPRESENTATION('{model_name}',(#{brep_id},#11),#12);"
    )

    total_lines.append(f"#{eid}=SHAPE_DEFINITION_REPRESENTATION(#7,#{shape_rep_id});")
    total_lines.append("ENDSEC;")
    total_lines.append("END-ISO-10303-21;")

    step_path.parent.mkdir(parents=True, exist_ok=True)
    step_path.write_text("\n".join(total_lines), encoding="ascii")
    print(
        f"  [+] Generated {step_path.name} ({step_path.stat().st_size:,} bytes, {len(point_map):,} unique vertices)"
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    for k in ["iter_like", "sparc_like", "diiid_like", "tcv_like"]:
        stl = root / "assets" / "3d" / "models" / f"{k}.stl"
        step = root / "assets" / "3d" / "models" / f"{k}.step"
        if stl.exists():
            stl_to_step(stl, step, k)
