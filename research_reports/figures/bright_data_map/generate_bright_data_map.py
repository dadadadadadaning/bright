"""Generate a publication-ready map of BRIGHT disaster-event locations.

The event names and sample counts are derived from the repository's
``bda_benchmark/dataset/splitname/standard_ML`` split files. Coordinates are
representative event/city points (not per-tile geocentres).
"""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import shapefile


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "bda_benchmark" / "dataset" / "splitname" / "standard_ML"
WORLD_SHP = OUT_DIR / "naturalearth" / "ne_110m_admin_0_countries.shp"


# Representative points follow the event locations used by the BRIGHT dataset.
# Label offsets are in display points and are tuned for a 16:8.5 canvas.
EVENTS = [
    dict(key="marshall-wildfire", place="Colorado, USA", date="Dec 30, 2021", lat=39.96, lon=-105.17, kind="Wildfire", cause="Natural", off=(-15, 21), ha="left"),
    dict(key="hawaii-wildfire", place="Maui, Hawaii, USA", date="Aug 08, 2023", lat=20.80, lon=-156.33, kind="Wildfire", cause="Natural", off=(5, 17), ha="left"),
    dict(key="mexico-hurricane", place="Acapulco, Mexico", date="Oct 25, 2023", lat=16.85, lon=-99.91, kind="Hurricane", cause="Natural", off=(-43, 14), ha="left"),
    dict(key="haiti-earthquake", place="Les Cayes, Haiti", date="Aug 14, 2021", lat=18.20, lon=-73.75, kind="Earthquake", cause="Natural", off=(7, 13), ha="left"),
    dict(key="la_palma-volcano", place="La Palma, Spain", date="Sep 19, 2021", lat=28.68, lon=-17.86, kind="Volcano eruption", cause="Natural", off=(-43, -19), ha="left"),
    dict(key="morocco-earthquake", place="Morocco", date="Sep 08, 2023", lat=31.07, lon=-8.41, kind="Earthquake", cause="Natural", off=(-8, 19), ha="left"),
    dict(key="bata-explosion", place="Bata, Equatorial Guinea", date="Mar 07, 2021", lat=1.86, lon=9.77, kind="Explosion", cause="Man-made", off=(-56, -18), ha="left"),
    dict(key="congo-volcano", place="Goma, DR Congo", date="May 22, 2021", lat=-1.66, lon=29.22, kind="Volcano eruption", cause="Natural", off=(12, -17), ha="left"),
    dict(key="libya-flood", place="Derna, Libya", date="Sep 11, 2023", lat=32.77, lon=22.64, kind="Flood", cause="Natural", off=(-56, -22), ha="left"),
    dict(key="beirut-explosion", place="Beirut, Lebanon", date="Aug 04, 2020", lat=33.89, lon=35.50, kind="Explosion", cause="Man-made", off=(-3, -25), ha="left"),
    dict(key="turkey-earthquake", place="Türkiye", date="Feb 06, 2023", lat=37.17, lon=37.03, kind="Earthquake", cause="Natural", off=(10, -2), ha="left"),
    dict(key="ukraine-conflict", place="Ukraine", date="2022", lat=48.40, lon=37.80, kind="Armed conflict", cause="Man-made", off=(9, 14), ha="left"),
    dict(key="myanmar-hurricane", place="Myanmar", date="May 14, 2023", lat=20.15, lon=92.90, kind="Hurricane", cause="Natural", off=(8, 13), ha="left"),
    dict(key="noto-earthquake", place="Noto, Japan", date="Jan 01, 2024", lat=37.50, lon=137.30, kind="Earthquake", cause="Natural", off=(10, -11), ha="left"),
]


COLORS = {
    "Earthquake": "#E69F00",
    "Flood": "#0072B2",
    "Hurricane": "#56B4E9",
    "Volcano eruption": "#D55E00",
    "Wildfire": "#009E73",
    "Explosion": "#CC79A7",
    "Armed conflict": "#6A3D9A",
}

ABBREVIATIONS = {
    "Earthquake": "EQ",
    "Flood": "FL",
    "Hurricane": "HU",
    "Volcano eruption": "VO",
    "Wildfire": "WF",
    "Explosion": "EX",
    "Armed conflict": "AC",
}


def read_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    pattern = re.compile(r"^(.+?)_\d+$")
    for path in sorted(SPLIT_DIR.glob("*_set.txt")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            match = pattern.match(raw.strip())
            if match:
                counts[match.group(1)] += 1
    return counts


def draw_world(ax: plt.Axes) -> None:
    reader = shapefile.Reader(str(WORLD_SHP), encoding="utf-8")
    for shape in reader.shapes():
        pts = shape.points
        parts = list(shape.parts) + [len(pts)]
        for start, end in zip(parts[:-1], parts[1:]):
            segment = pts[start:end]
            if len(segment) < 3:
                continue
            xs, ys = zip(*segment)
            # Identical fill with no polygon edges removes political/country
            # borders while preserving a clean land-versus-ocean silhouette.
            ax.fill(xs, ys, facecolor="#E5E9E3", edgecolor="none", linewidth=0, zorder=0)


def marker_size(count: int) -> float:
    # Matplotlib scatter sizes are areas in pt^2. A square-root count mapping
    # keeps the largest event visible without overwhelming nearby labels.
    return 72 + 3.25 * count ** 0.5


def format_lon(value: int) -> str:
    if value == 0:
        return "0°"
    return f"{abs(value)}° {'W' if value < 0 else 'E'}"


def format_lat(value: int) -> str:
    if value == 0:
        return "0°"
    return f"{abs(value)}° {'S' if value < 0 else 'N'}"


def main() -> None:
    counts = read_counts()
    missing = sorted({event["key"] for event in EVENTS} - counts.keys())
    if missing:
        raise RuntimeError(f"Events missing from standard split files: {missing}")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(16, 8.5), constrained_layout=False)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.95, bottom=0.095)
    draw_world(ax)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#F8FBFD")

    xticks = list(range(-180, 181, 30))
    yticks = list(range(-60, 91, 30))
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([format_lon(x) for x in xticks])
    ax.set_yticklabels([format_lat(y) for y in yticks])
    ax.tick_params(
        axis="x", which="both", labeltop=True, labelbottom=False,
        top=True, bottom=True, direction="in", pad=3, length=4,
    )
    ax.tick_params(
        axis="y", which="both", labelright=True, labelleft=False,
        right=True, left=True, direction="in", pad=3, length=4,
    )
    ax.grid(color="#7C858B", linewidth=0.55, alpha=0.42, zorder=-1)

    for event in EVENTS:
        count = counts[event["key"]]
        # Use one symbol for every event. Disaster type is encoded by colour;
        # this avoids the earlier star/diamond distinction being mistaken for
        # two different kinds of plotted objects.
        marker = "*"
        ax.scatter(
            event["lon"], event["lat"],
            s=marker_size(count), marker=marker,
            c=COLORS[event["kind"]], edgecolors="white", linewidths=0.85,
            zorder=5,
        )
        label = (
            f"{event['place']}\n"
            f"{event['date']}  ·  {ABBREVIATIONS[event['kind']]}  ·  n={count:,}"
        )
        ax.annotate(
            label,
            xy=(event["lon"], event["lat"]),
            xytext=event["off"], textcoords="offset points",
            ha=event["ha"], va="center",
            fontsize=7.8, fontweight="bold", color="#20252A",
            linespacing=1.14,
            arrowprops=dict(
                arrowstyle="-", color="#677078", lw=0.55,
                shrinkA=1.5, shrinkB=4.5,
            ),
            bbox=dict(
                boxstyle="round,pad=0.18", facecolor="white",
                edgecolor="none", alpha=0.74,
            ),
            zorder=6,
        )

    type_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", label=kind,
            markerfacecolor=color, markeredgecolor="white", markersize=7,
        )
        for kind, color in COLORS.items()
    ]
    type_legend = ax.legend(
        handles=type_handles, title="Disaster type", ncol=1,
        loc="lower left", bbox_to_anchor=(0.012, 0.022),
        frameon=True, fancybox=False, framealpha=0.94,
        facecolor="white", edgecolor="#4B5157",
        borderpad=0.65, labelspacing=0.32, handletextpad=0.45,
        fontsize=7.8, title_fontsize=8.4,
    )
    ax.add_artist(type_legend)

    symbol_handles = [
        Line2D([0], [0], marker="*", color="none", label="Event location",
               markerfacecolor="#333333", markeredgecolor="white", markersize=10),
    ]
    symbol_legend = ax.legend(
        handles=symbol_handles, title="Map symbol", ncol=1,
        loc="lower right", bbox_to_anchor=(0.988, 0.022),
        frameon=True, fancybox=False, framealpha=0.94,
        facecolor="white", edgecolor="#4B5157",
        borderpad=0.65, labelspacing=0.45, handletextpad=0.45,
        fontsize=7.8, title_fontsize=8.4,
    )
    ax.add_artist(symbol_legend)

    total = sum(counts.values())
    ax.text(
        0.5, -0.064,
        f"BRIGHT standard split: 14 disaster events · {total:,} paired samples · all locations use star symbols",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=8.3, color="#454D54",
    )
    ax.text(
        0.5, -0.096,
        "Representative event/city coordinates are shown; points do not denote every image-tile footprint.",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=7.4, color="#69737B",
    )

    for spine in ax.spines.values():
        spine.set_color("#343A40")
        spine.set_linewidth(0.9)

    stem = OUT_DIR / "BRIGHT_data_geographic_distribution_no_country_borders"
    fig.savefig(stem.with_suffix(".png"), dpi=320, facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # Clean basemap used by the editable PowerPoint version. All semantic
    # content (stars, labels, legends and footer) is added as native slide
    # objects there.
    base_fig, base_ax = plt.subplots(figsize=(16, 7.2))
    base_fig.subplots_adjust(left=0.025, right=0.975, top=0.96, bottom=0.08)
    draw_world(base_ax)
    base_ax.set_xlim(-180, 180)
    base_ax.set_ylim(-60, 85)
    base_ax.set_aspect("equal", adjustable="box")
    base_ax.set_facecolor("#F8FBFD")
    base_ax.set_xticks(list(range(-180, 181, 30)))
    base_ax.set_yticks(list(range(-60, 91, 30)))
    base_ax.set_xticklabels([])
    base_ax.set_yticklabels([])
    base_ax.tick_params(direction="in", length=3)
    base_ax.grid(color="#7C858B", linewidth=0.55, alpha=0.42, zorder=-1)
    for spine in base_ax.spines.values():
        spine.set_color("#343A40")
        spine.set_linewidth(0.9)
    base_fig.savefig(
        OUT_DIR / "BRIGHT_world_base_no_country_borders_for_ppt.png",
        dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.02,
    )
    plt.close(base_fig)

    with (OUT_DIR / "BRIGHT_event_locations.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "event", "location", "date", "latitude", "longitude",
                "disaster_type", "cause", "sample_count",
            ],
        )
        writer.writeheader()
        for event in EVENTS:
            writer.writerow(
                {
                    "event": event["key"],
                    "location": event["place"],
                    "date": event["date"],
                    "latitude": event["lat"],
                    "longitude": event["lon"],
                    "disaster_type": event["kind"],
                    "cause": event["cause"],
                    "sample_count": counts[event["key"]],
                }
            )


if __name__ == "__main__":
    main()
