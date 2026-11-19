"""Plot survival time, walking distance and distance from origin against training step
(PPO update, the wandb step) and against environment steps, for the
policy-architecture runs in Rajan_Lab_MARL/haldun-forageworld, up to 30k updates.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "arch_comparison")
CACHE = os.path.join(OUTDIR, "wandb_history_cache.npz")
RUNS = [  # (cache key, wandb run id, legend label)
    ("gru", "rqgq6aef", "GRU"),
    ("agalite", "vw2osu7f", "AGaLiTe"),
    ("mlp", "5oeqkh8d", "MLP (no memory)"),
    ("gtrxl-64", "ong2l8gd", "GTrXL (memory 64)"),
    ("gtrxl-128", "vhainzeq", "GTrXL (memory 128)"),
    ("gtrxl-256", "9cwqc6ar", "GTrXL (memory 256)"),
    ("gtrxl-512", "agsd5hzh", "GTrXL (memory 512)"),
]
METRICS = [  # (wandb key, y label, file name stem)
    ("overview/episode_length", "Survival Time (Timesteps)", "survival_time"),
    ("overview/walking_distance", "Total Walking Distance", "walking_distance"),
    ("overview/distance_to_spawn", "Average Distance from Origin", "distance_from_origin"),
]
X_AXES = [  # (x label, env steps per training step, file name suffix)
    ("Training Step", 1, "training_step"),
    ("Environment Steps", 512 * 64, "env_steps"),  # NUM_ENVS x NUM_STEPS per PPO update
]
# Legend columns, left to right, in two rows; GRU sits alone, centered between the rows.
LEGEND_COLUMNS = [["agalite", "mlp"], ["gru"], ["gtrxl-64", "gtrxl-128"], ["gtrxl-256", "gtrxl-512"]]
WINDOW = 1000  # moving average, in PPO updates
BAND = 0.5  # shade mean +- BAND * std of the updates in the window (one seed per model)
STOP_AT_UPDATE = 30000


def load_history():
    if os.path.exists(CACHE):
        with np.load(CACHE) as cached:
            return {key: cached[key] for key in cached.files}
    import wandb

    api = wandb.Api(timeout=180)
    keys = [key for key, _, _ in METRICS]
    history = {}
    for name, run_id, _ in RUNS:
        # scan_history fails on these runs with wandb 0.28; this returns 10k evenly spaced rows.
        rows = api.run(f"Rajan_Lab_MARL/haldun-forageworld/{run_id}").history(samples=10000, keys=keys, pandas=False)
        table = np.array([[row["_step"]] + [row[k] for k in keys] for row in rows], dtype=float)
        history[name] = table[np.argsort(table[:, 0])]
    os.makedirs(OUTDIR, exist_ok=True)
    np.savez_compressed(CACHE, **history)
    return history


def moving_mean_std(values, steps):
    """Centered mean and std over WINDOW updates; the window shrinks at the ends."""
    half = WINDOW / 2
    lo = np.searchsorted(steps, steps - half, side="left")
    hi = np.searchsorted(steps, steps + half, side="right")
    n = hi - lo
    s1 = np.concatenate([[0.0], np.cumsum(values)])
    s2 = np.concatenate([[0.0], np.cumsum(values ** 2)])
    mean = (s1[hi] - s1[lo]) / n
    return mean, np.sqrt(np.maximum((s2[hi] - s2[lo]) / n - mean ** 2, 0.0))


def legend_below(fig, lines):
    """Boxed legend along the bottom of the figure, laid out as LEGEND_COLUMNS.

    plt.legend fills columns top-down, so a lone entry can't be centered
    between the rows; this builds the same box from offsetbox pieces."""
    size = plt.rcParams["font.size"]

    def entry(line):
        handle = DrawingArea(1.6 * size, size)
        handle.add_artist(Line2D([0, 1.6 * size], [size / 2] * 2, color=line.get_color(), lw=line.get_linewidth()))
        return HPacker(children=[handle, TextArea(line.get_label())], align="center", pad=0, sep=0.6 * size)

    columns = [VPacker(children=[entry(lines[key]) for key in keys], align="left", pad=0, sep=0.5 * size)
               for keys in LEGEND_COLUMNS]
    box = AnchoredOffsetbox(loc="lower center", child=HPacker(children=columns, align="center", pad=0, sep=1.4 * size),
                            bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure, frameon=True, pad=0.5)
    box.patch.set_boxstyle("round,pad=0.2")
    box.patch.set_edgecolor("0.8")
    fig.add_artist(box)


def main():
    history = load_history()
    plt.rcParams["font.size"] *= 2  # 2x the default text size
    for xlabel, scale, suffix in X_AXES:
        for i, (_key, ylabel, stem) in enumerate(METRICS):
            fig, ax = plt.subplots(figsize=(14, 8))  # room for the 2x text and the legend underneath
            lines = {}
            for name, _run_id, label in RUNS:
                table = history[name]
                table = table[table[:, 0] < STOP_AT_UPDATE]
                steps = table[:, 0]
                x = steps * scale
                mean, std = moving_mean_std(table[:, 1 + i], steps)
                lines[name], = ax.plot(x, mean, label=label)
                ax.fill_between(x, mean - BAND * std, mean + BAND * std, color=lines[name].get_color(), alpha=0.2, lw=0)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y")
            ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0), useMathText=True)
            legend_below(fig, lines)
            fig.tight_layout(rect=(0, 0.17, 1, 1))  # keep the bottom 17% for the legend
            fig.savefig(os.path.join(OUTDIR, f"{stem}_{suffix}.png"), dpi=200)
            plt.close(fig)


if __name__ == "__main__":
    main()
