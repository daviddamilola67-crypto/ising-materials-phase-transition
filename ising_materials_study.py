#!/usr/bin/env python3
"""Monte Carlo study of magnetic ordering in a 2D Ising material model.

The square-lattice nearest-neighbor Ising model illustrates temperature-
dependent ordering and finite-size effects. It is not a quantitative model
of a specific real material.
"""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SIZES = (16, 32)
TEMPERATURES = np.round(np.arange(1.8, 3.21, 0.1), 2)
J = 1.0
KB = 1.0
EQUILIBRATION_SWEEPS = 500
MEASUREMENT_SWEEPS = 1200
SAMPLE_EVERY = 2
SEED = 20261026


def metropolis_sweep(spins, temperature, rng, even, odd):
    """Update both checkerboard sublattices with Metropolis steps."""
    for mask in (even, odd):
        neighbors = (
            np.roll(spins, 1, axis=0) + np.roll(spins, -1, axis=0)
            + np.roll(spins, 1, axis=1) + np.roll(spins, -1, axis=1)
        )
        delta_energy = 2.0 * J * spins * neighbors
        accept = (delta_energy <= 0.0) | (
            rng.random(spins.shape) < np.exp(-delta_energy / (KB * temperature))
        )
        spins[mask & accept] *= -1


def measure(spins):
    """Return total energy and signed magnetization per lattice site."""
    right, down = np.roll(spins, -1, axis=1), np.roll(spins, -1, axis=0)
    energy_per_site = -J * np.mean(spins * (right + down))
    return float(energy_per_site), float(np.mean(spins))


def simulate(size, temperature, rng):
    if size % 2:
        raise ValueError("Checkerboard updates require an even lattice size.")
    rows, cols = np.indices((size, size))
    even = (rows + cols) % 2 == 0
    odd = ~even
    spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(size, size))
    for _ in range(EQUILIBRATION_SWEEPS):
        metropolis_sweep(spins, temperature, rng, even, odd)

    energies, magnetizations = [], []
    for sweep in range(MEASUREMENT_SWEEPS):
        metropolis_sweep(spins, temperature, rng, even, odd)
        if (sweep + 1) % SAMPLE_EVERY == 0:
            energy, magnetization = measure(spins)
            energies.append(energy)
            magnetizations.append(magnetization)

    energy, magnetization = np.asarray(energies), np.asarray(magnetizations)
    sites = size * size
    abs_m = np.abs(magnetization)
    heat_capacity = sites * np.var(energy, ddof=1) / (KB * temperature**2)
    susceptibility_abs = sites * (
        np.mean(magnetization**2) - np.mean(abs_m)**2
    ) / (KB * temperature)
    return {
        "lattice_size": size,
        "temperature_over_J_per_kB": float(temperature),
        "mean_energy_per_site_over_J": float(np.mean(energy)),
        "mean_absolute_magnetization_per_site": float(np.mean(abs_m)),
        "heat_capacity_per_site_over_kB": float(heat_capacity / KB),
        "abs_magnetization_susceptibility_per_site": float(susceptibility_abs),
        "samples": int(len(energy)),
    }


def write_svg(path, rows):
    """Write heat-capacity and magnetization curves as a plain SVG."""
    width, height, left, right, top = 920, 620, 85, 150, 55
    panel_height, gap = 205, 65
    plot_width = width - left - right
    colors = {16: "#1769aa", 32: "#d04a35"}
    panels = [
        ("heat_capacity_per_site_over_kB", "Heat capacity per site / kB"),
        ("mean_absolute_magnetization_per_site", "Mean absolute magnetization per site"),
    ]
    temps = sorted({r["temperature_over_J_per_kB"] for r in rows})
    x_min, x_max = min(temps), max(temps)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="460" y="28" text-anchor="middle" font-family="Arial" font-size="19" font-weight="bold">2D Ising model: finite-size temperature sweep</text>']
    for panel, (key, label) in enumerate(panels):
        y0 = top + panel * (panel_height + gap)
        ymax = max(max(r[key] for r in rows) * 1.12, 0.1)
        for tick in range(5):
            val = ymax * tick / 4
            yy = y0 + panel_height - val / ymax * panel_height
            svg.extend([
                f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_width}" y2="{yy:.1f}" stroke="#e2e6ea"/>',
                f'<text x="{left-12}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{val:.2f}</text>',
            ])
        svg.append(f'<text x="{left}" y="{y0-12}" font-family="Arial" font-size="15" font-weight="bold">{label}</text>')
        for size in SIZES:
            series = sorted((r for r in rows if r["lattice_size"] == size),
                            key=lambda r: r["temperature_over_J_per_kB"])
            points = []
            for row in series:
                x = left + (row["temperature_over_J_per_kB"] - x_min) / (x_max - x_min) * plot_width
                y = y0 + panel_height - row[key] / ymax * panel_height
                points.append(f"{x:.1f},{y:.1f}")
            svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[size]}" stroke-width="2.5"/>')
        svg.extend([
            f'<line x1="{left}" y1="{y0+panel_height}" x2="{left+plot_width}" y2="{y0+panel_height}" stroke="#333"/>',
            f'<line x1="{left}" y1="{y0}" x2="{left}" y2="{y0+panel_height}" stroke="#333"/>',
        ])
    for size, color in colors.items():
        y = top + 55 + (size - 16) * 2
        svg.extend([
            f'<line x1="{width-130}" y1="{y}" x2="{width-108}" y2="{y}" stroke="{color}" stroke-width="3"/>',
            f'<text x="{width-100}" y="{y+4}" font-family="Arial" font-size="13">L = {size}</text>',
        ])
    svg.append(f'<text x="460" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="14">Temperature (J / kB)</text>')
    svg.append('</svg>')
    path.write_text(chr(10).join(svg) + chr(10), encoding="utf-8")


def main():
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = [simulate(size, float(temperature), rng)
            for size in SIZES for temperature in TEMPERATURES]
    with (RESULTS / "ising_temperature_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    peaks = {}
    for size in SIZES:
        size_rows = [r for r in rows if r["lattice_size"] == size]
        peak = max(size_rows, key=lambda r: r["heat_capacity_per_site_over_kB"])
        peaks[str(size)] = {
            "sampled_heat_capacity_peak_temperature_J_per_kB": peak["temperature_over_J_per_kB"],
            "peak_heat_capacity_per_site_over_kB": peak["heat_capacity_per_site_over_kB"],
        }
    summary = {
        "model": "2D square-lattice nearest-neighbor Ising ferromagnet",
        "units": "Reduced units: J = 1 and kB = 1",
        "lattice_sizes": list(SIZES),
        "temperature_range_J_per_kB": [float(TEMPERATURES[0]), float(TEMPERATURES[-1])],
        "equilibration_sweeps_per_temperature": EQUILIBRATION_SWEEPS,
        "measurement_sweeps_per_temperature": MEASUREMENT_SWEEPS,
        "sampling_interval_sweeps": SAMPLE_EVERY,
        "random_seed": SEED,
        "sampled_heat_capacity_peaks": peaks,
        "reference": "The infinite square-lattice model has kB*T/J = 2/ln(1+sqrt(2)) ≈ 2.269; finite-size sampled peaks need not equal this value.",
        "limitations": [
            "Synthetic pedagogical model, not a calibrated simulation of a named material.",
            "Finite lattices and finite Monte Carlo sampling introduce size effects and statistical uncertainty.",
            "The reported absolute-magnetization susceptibility uses the stated finite-size estimator.",
            "No error bars, magnetic anisotropy, long-range interactions, or quantum effects are included.",
        ],
    }
    (RESULTS / "study_summary.json").write_text(json.dumps(summary, indent=2) + chr(10), encoding="utf-8")
    write_svg(RESULTS / "ising_curves.svg", rows)
    print("Computed", len(rows), "size-temperature points.")
    print("Sampled heat-capacity peak temperatures (J/kB):", peaks)
    print("Results saved in", RESULTS)


if __name__ == "__main__":
    main()
