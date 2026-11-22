"""Render the planning map to nav/reports/map.png (occ / cliff / cost / rooms)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nav.mapping import build_map  # noqa: E402


def main(out=None, sigma=0.0):
    gm = build_map(lateral_sigma=sigma)
    out = Path(out) if out else Path(__file__).resolve().parents[1] / "reports" / "map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = gm.bounds
    ext = (x0, x1, y0, y1)
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.2))
    base = np.ones(gm.occ.shape + (3,))
    base[gm.occ] = (0.1, 0.1, 0.1)
    base[gm.cliff] = (0.85, 0.15, 0.15)
    ax[0].imshow(base, origin="lower", extent=ext)
    ax[0].set_title("occupancy (black) / cliff (red)")
    c = np.where(np.isinf(gm.cost), np.nan, gm.cost)
    im = ax[1].imshow(c, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1)
    ax[1].imshow(np.where(gm.lethal, 1.0, np.nan), origin="lower", extent=ext, cmap="gray_r", vmin=0, vmax=1.6)
    ax[1].set_title(f"cost (lethal = dark), r_inflate={gm.params['r_inflate']:.3f} m")
    fig.colorbar(im, ax=ax[1], fraction=0.04)
    ax[2].imshow(base, origin="lower", extent=ext, alpha=0.35)
    for k, (nm, (xa, ya, xb, yb)) in enumerate(gm.rooms.items()):
        ax[2].add_patch(plt.Rectangle((xa, ya), xb - xa, yb - ya, fill=False, ec=f"C{k}", lw=2))
        ax[2].text((xa + xb) / 2, (ya + yb) / 2, nm, ha="center", color=f"C{k}", fontsize=11, weight="bold")
        for (px, py) in gm.sample_free(nm, 5):
            ax[2].plot(px, py, "o", color=f"C{k}", ms=4)
    for dw in gm.doorways:
        ax[2].plot([dw["p0"][0], dw["p1"][0]], [dw["p0"][1], dw["p1"][1]], "-", color="orange", lw=4)
    ax[2].set_title("rooms, doorways (orange), free samples")
    for a in ax:
        a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print("saved", out)


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
