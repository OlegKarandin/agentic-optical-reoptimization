"""Offline, one-shot: clip and simplify Natural Earth's land polygons into
the small land/sea backdrop the run viewer draws under the plant.

Not part of any pipeline -- the output (`viewer_land.json`) is checked in
and `build_viewer_data.py` only reads it. Rerun only to change the clip box
or the simplification tolerance.

Source: Natural Earth 1:50m physical land (public domain),
https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_50m_land.geojson
Download it once and pass it as --source; this script never fetches.

Land only, dissolved: no country borders, by design -- the backdrop is there
to separate land from sea, not to carry political geography.

Run with the storm-reoptimizer env (needs shapely):
    python tools/build_coastline.py --source ne_50m_land.geojson
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union

OUT_PATH = (Path(__file__).parent.parent / "src" / "storm_reoptimizer"
            / "data" / "viewer_land.json")

# Much wider than the viewer's own BBOX (lat 8-35, lon 68-97): with
# preserveAspectRatio="xMidYMid meet" a wide map pane shows well past the
# 800x800 box on either side, and panning can go further still. Land has to
# keep going there or the coast ends in a visible straight cut.
CLIP = {"lon_min": 45.0, "lon_max": 120.0, "lat_min": -12.0, "lat_max": 50.0}
TOLERANCE_DEG = 0.03
MIN_AREA_DEG2 = 0.02  # drops specks that would render as sub-pixel dots
DECIMALS = 2          # ~1 km -- far below one viewer pixel at full view


def _rings(geom) -> list[list[list[float]]]:
    """Exterior rings only, as [lat, lon] -- the viewer's own point order.
    Holes are dropped: the only ones in range are inland lakes the 1:50m
    land layer cuts out, and the backdrop does not need them."""
    polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
    out = []
    for p in polys:
        if p.area < MIN_AREA_DEG2:
            continue
        out.append([[round(lat, DECIMALS), round(lon, DECIMALS)]
                    for lon, lat in p.exterior.coords])
    return out


def build(source: Path) -> dict:
    raw = json.loads(source.read_text(encoding="utf-8"))
    clip = box(CLIP["lon_min"], CLIP["lat_min"],
               CLIP["lon_max"], CLIP["lat_max"])
    land = unary_union([shape(f["geometry"]) for f in raw["features"]])
    land = land.intersection(clip).simplify(TOLERANCE_DEG,
                                            preserve_topology=True)
    return {
        "source": "Natural Earth 1:50m land (public domain), clipped to "
                  f"lon {CLIP['lon_min']}..{CLIP['lon_max']}, lat "
                  f"{CLIP['lat_min']}..{CLIP['lat_max']}, simplified at "
                  f"{TOLERANCE_DEG} deg by tools/build_coastline.py",
        "rings": _rings(land),
    }


def main() -> None:
    p = argparse.ArgumentParser(prog="build_coastline",
                                description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True,
                   help="Path to a downloaded ne_50m_land.geojson.")
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()
    data = build(Path(args.source))
    out = Path(args.out)
    out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    points = sum(len(r) for r in data["rings"])
    print(f"wrote {out}: {len(data['rings'])} rings, {points} points, "
          f"{out.stat().st_size // 1024} KB", flush=True)


if __name__ == "__main__":
    main()
