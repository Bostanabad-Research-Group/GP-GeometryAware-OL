"""
Collects the saved results of all UQ methods and produces:
  1. A results table (mean ± std over seeds) in Markdown and LaTeX.
  2. A table of metrics per geometry-novelty quartile (Q1-Q4), if MMD scores were computed.
  3. One figure per dataset and test sample: ground truth, prediction, absolute error and the
     predicted standard deviation of every UQ method.

It only reads the files written by the UQ scripts (benchmarks_eval_UQ_<Method>.yaml and
<Method>_benchmark_data.pt), so it needs no GPU. Methods that were not run are skipped.

Usage:
    python olgp/compare_results.py                      # all datasets found
    python olgp/compare_results.py --problems naca car  # a subset
"""
import argparse
import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROBLEMS = {  # config name -> (folder name, label, spatial dim)
    "elasticity": ("Elasticity", "Elasticity", 2),
    "naca": ("Naca", "Airfoil", 2),
    "pipe": ("Pipe", "Pipe", 2),
    "car": ("Car", "ShapeNet Car", 3),
    "ahmed": ("Ahmed", "Ahmed Body", 3),
}
METHODS = {  # saved name -> (ablation folder, label)
    "Ensemble": ("ablation_standard", "Deep Ensemble"),
    "MCDropout": ("ablation_dropout", "MC Dropout"),
    "PNO": ("ablation_pno", "PNO"),
    "Perturbation": ("ablation_standard", "Perturbation"),
    "LA": ("ablation_standard", "LUNO-LA"),
    "REEFGP": ("ablation_standard", "REEF-GP"),
}
METRICS = {"rL2": "rL2", "nll": "NLL", "crps": "CRPS", "is": "NIS", "energy": "ES"}  # yaml key -> label


def default_out_root():
    """Reads base_path from configs/machine/local.yaml (set by scripts/fix_paths.sh)."""
    with open(os.path.join(REPO_ROOT, "configs", "machine", "local.yaml")) as f:
        base_path = yaml.safe_load(f)["base_path"]
    if "<ROOT_FOLDER>" in base_path:
        base_path = REPO_ROOT
    return os.path.join(base_path, "Out", "Baselines", "transolver")


def find_latest(pattern):
    """Most recently written file matching a recursive glob pattern, or None."""
    files = glob.glob(pattern, recursive=True)
    return max(files, key=os.path.getmtime) if files else None


# ==========================================
# Loading
# ==========================================
def load_problem_results(problem_dir):
    """Returns {method: {"metrics": {...}, "quantiles": {...}}} including the deterministic Base."""
    results = {}

    # Base: deterministic Transolver, rL2 averaged over seeds
    base_rl2 = []
    for path in glob.glob(os.path.join(problem_dir, "ablation_standard", "N_*", "seed_*", "Benchmarks", "benchmarks_eval.yaml")):
        with open(path) as f:
            base_rl2.append(yaml.safe_load(f)["metrics"]["rel_l2"])
    if base_rl2:
        results["Base"] = {"metrics": {"rL2": (np.mean(base_rl2), np.std(base_rl2))}, "quantiles": {}}

    for method, (ablation, _) in METHODS.items():
        path = find_latest(os.path.join(problem_dir, ablation, "**", "Benchmarks", f"benchmarks_eval_UQ_{method}.yaml"))
        if path is None:
            continue
        with open(path) as f:
            metrics = yaml.safe_load(f)["metrics"]
        results[method] = {
            "metrics": {k: (metrics[k]["mean"], metrics[k]["std"]) for k in METRICS if k in metrics},
            "quantiles": {
                q: {k: (v["mean"], v["std"]) for k, v in q_metrics.items()}
                for q, q_metrics in metrics.get("quantiles", {}).items()
            },
        }
    return results


def load_plot_data(problem_dir, method):
    ablation, _ = METHODS[method]
    path = find_latest(os.path.join(problem_dir, ablation, "**", "Benchmarks", f"{method}_benchmark_data.pt"))
    return torch.load(path, map_location="cpu", weights_only=False) if path else None


# ==========================================
# Tables
# ==========================================
def column_scale(values):
    """Power of 10 used to print a column (e.g. CRPS x 1e-4), 0 if no scaling is needed."""
    values = [abs(v) for v in values if v is not None]
    if not values or (max(values) < 1000 and min(values) >= 0.1):
        return 0
    exponent = int(math.floor(math.log10(max(values))))
    return exponent if max(values) >= 1000 else exponent - 1


def format_cell(mean_std, scale, rank, latex):
    mean, std = mean_std
    text = f"{mean / 10 ** scale:.2f}"
    if std:  # the Deep Ensemble is a single model set, so it has no std over seeds
        text += (r" \pm " if latex else " ± ") + f"{std / 10 ** scale:.2f}"
    if latex:
        text = {"best": rf"\mathbf{{{text}}}", "second": rf"\underline{{{text}}}"}.get(rank, text)
        return f"${text}$"
    return {"best": f"**{text}**", "second": f"_{text}_"}.get(rank, text)


def results_table(all_results, latex):
    """One block per dataset, one row per method. Best in bold, second best underlined (Base excluded)."""
    lines = []
    if latex:
        lines += [r"\begin{tabular}{l" + "c" * len(METRICS) + "}", r"\toprule",
                  "Method & " + " & ".join(rf"{m} $\downarrow$" for m in METRICS.values()) + r" \\"]
    else:
        lines += ["| Method | " + " | ".join(METRICS.values()) + " |", "|---" * (len(METRICS) + 1) + "|"]

    for problem, results in all_results.items():
        scales, ranks = {}, {}
        for key in METRICS:
            uq_values = {m: r["metrics"][key][0] for m, r in results.items() if m != "Base" and key in r["metrics"]}
            scales[key] = -2 if key == "rL2" else column_scale(uq_values.values())  # rL2 in %
            ordered = sorted(uq_values, key=uq_values.get)
            ranks[key] = {m: rank for m, rank in zip(ordered, ["best", "second"])}

        units = ["(\\%)" if latex else "(%)"] + [
            (rf"($\times 10^{{{scales[k]}}}$)" if latex else f"(x1e{scales[k]})") if scales[k] else "" for k in list(METRICS)[1:]
        ]
        if latex:
            lines += [r"\midrule", rf"\emph{{{PROBLEMS[problem][1]}}} & " + " & ".join(units) + r" \\", r"\midrule"]
        else:
            lines.append(f"| **{PROBLEMS[problem][1]}** | " + " | ".join(units) + " |")

        for method in ["Base"] + list(METHODS):
            if method not in results:
                continue
            label = "Base (no UQ)" if method == "Base" else METHODS[method][1]
            cells = [
                format_cell(results[method]["metrics"][k], scales[k], ranks[k].get(method), latex)
                if k in results[method]["metrics"] else "--"
                for k in METRICS
            ]
            lines.append((f"{label} & " + " & ".join(cells) + r" \\") if latex else f"| {label} | " + " | ".join(cells) + " |")

    if latex:
        lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def quantile_table(all_results):
    """rL2 and NLL per geometry-novelty quartile (Q1: closest to training, Q4: most novel)."""
    lines = ["| Dataset | Method | " + " | ".join(f"{q} rL2 (%) | {q} NLL" for q in ["Q1", "Q2", "Q3", "Q4"]) + " |",
             "|---" * 10 + "|"]
    for problem, results in all_results.items():
        for method, r in results.items():
            if not r["quantiles"]:
                continue
            cells = []
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                cells += [format_cell(r["quantiles"][q]["rL2"], -2, None, latex=False),
                          format_cell(r["quantiles"][q]["nll"], 0, None, latex=False)]
            lines.append(f"| {PROBLEMS[problem][1]} | {METHODS[method][1]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) if len(lines) > 2 else ""


# ==========================================
# Figures
# ==========================================
def scatter_field(fig, ax, pts, values, title, cmap, vmin=None, vmax=None):
    if pts.shape[-1] == 3:
        sc = ax.scatter(pts[:, 0], -pts[:, 1], pts[:, 2], c=values, s=3, cmap=cmap, vmin=vmin, vmax=vmax,
                        edgecolors="none", rasterized=True)
        ax.set_box_aspect(np.ptp(pts, axis=0))
        ax.view_init(elev=25, azim=45)
    else:
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=values, s=5, cmap=cmap, vmin=vmin, vmax=vmax,
                        edgecolors="none", rasterized=True)
        ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, pad=2, fontsize=8)
    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08, fraction=0.06)
    cbar.ax.tick_params(labelsize=6)
    cbar.locator = plt.MaxNLocator(nbins=3)
    cbar.update_ticks()


def sample_fields(data, key, sample, n_points):
    """Field of seed 0 for one of the 3 stored test samples (25th/50th/75th error percentile)."""
    values = np.asarray(data["global"][key]).reshape(-1, len(data["global"]["indices"]), n_points)
    return values[0, sample]


def plot_samples(problem, problem_dir, out_dir, reference):
    """For each stored sample: truth, prediction and error of `reference`, then the std of every method."""
    plot_data = {m: d for m in METHODS if (d := load_plot_data(problem_dir, m)) is not None}
    if not plot_data:
        return
    reference = reference if reference in plot_data else next(iter(plot_data))
    # All methods must show the same test samples (chosen from the seed-0 backbone's errors)
    ref_indices = [int(i) for i in plot_data[reference]["global"]["indices"]]
    for m in list(plot_data):
        indices = [int(i) for i in plot_data[m]["global"]["indices"]]
        if indices != ref_indices:
            print(f"WARNING: {METHODS[m][1]} stored different test samples ({indices} vs {ref_indices}), "
                  f"so it was run with other backbones. Skipping it in the {PROBLEMS[problem][1]} figures.")
            del plot_data[m]
    dim = PROBLEMS[problem][2]
    pts_all = np.asarray(plot_data[reference]["global"]["pts"])[0, ..., :dim]  # [3, N, dim], seed 0
    n_samples, n_points = pts_all.shape[:2]
    percentiles = ["25th", "50th", "75th"]

    for sample in range(n_samples):
        pts = pts_all[sample]
        gt = sample_fields(plot_data[reference], "gt", sample, n_points)
        pred = sample_fields(plot_data[reference], "pred", sample, n_points)
        panels = [("Ground truth", gt, "viridis", gt.min(), gt.max()),
                  (f"Prediction ({METHODS[reference][1]})", pred, "viridis", gt.min(), gt.max()),
                  ("Absolute error", np.abs(gt - pred), "magma", None, None)]
        panels += [(f"Std: {METHODS[m][1]}", sample_fields(d, "std", sample, n_points), "plasma", None, None)
                   for m, d in plot_data.items()]

        n_rows = math.ceil(len(panels) / 5)  # at most 5 panels per row, rows filled evenly
        n_cols = math.ceil(len(panels) / n_rows)
        fig = plt.figure(figsize=(2.0 * n_cols, 2.0 * n_rows))
        for i, (title, values, cmap, vmin, vmax) in enumerate(panels):
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d" if dim == 3 else None)
            scatter_field(fig, ax, pts, values, title, cmap, vmin, vmax)
        fig.suptitle(f"{PROBLEMS[problem][1]}: test sample at the {percentiles[sample]} percentile of base error", fontsize=9)
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.88, wspace=0.15, hspace=0.35)

        save_path = os.path.join(out_dir, f"{PROBLEMS[problem][0]}_sample_{percentiles[sample]}.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure: {save_path}")


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Compare the saved results of all UQ methods.")
    parser.add_argument("--problems", nargs="+", default=list(PROBLEMS), choices=list(PROBLEMS))
    parser.add_argument("--out-root", default=None, help="Results folder (default: <base_path>/Out/Baselines/transolver)")
    parser.add_argument("--reference", default="REEFGP", choices=list(METHODS),
                        help="Method whose mean is shown as the prediction in the figures")
    args = parser.parse_args()

    out_root = args.out_root or default_out_root()
    out_dir = os.path.join(out_root, "Comparison")
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    for problem in args.problems:
        problem_dir = os.path.join(out_root, PROBLEMS[problem][0])
        results = load_problem_results(problem_dir)
        if not results:
            print(f"No results found for {PROBLEMS[problem][1]} in {problem_dir}. Skipping.")
            continue
        all_results[problem] = results
        plot_samples(problem, problem_dir, out_dir, args.reference)

    if not all_results:
        return
    tables = {
        "results_table.md": results_table(all_results, latex=False),
        "results_table.tex": results_table(all_results, latex=True),
        "results_quartiles.md": quantile_table(all_results),
    }
    for filename, table in tables.items():
        if table:
            with open(os.path.join(out_dir, filename), "w") as f:
                f.write(table + "\n")
    print("\n" + tables["results_table.md"])
    if tables["results_quartiles.md"]:
        print("\nMetrics per geometry-novelty quartile (Q1: closest to training, Q4: most novel)\n")
        print(tables["results_quartiles.md"])
    print(f"\nTables and figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
