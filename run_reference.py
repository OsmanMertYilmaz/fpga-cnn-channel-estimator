"""Aşama 1 referansı: 72×14 ızgarada LS + interpolasyon ve LMMSE için NMSE-SNR eğrileri.

Çalıştırma:  .venv/bin/python run_reference.py
Çıktılar:    results/nmse_reference.png, results/nmse_reference.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from estimators import channel_covariance, lmmse, lmmse_matrix, ls_interp, ls_pilots, nmse_db
from ofdm_channel import GridConfig, generate_channels, qpsk_pilots, received_pilots

PROFILES = ("TDL-A", "TDL-B", "TDL-C")
DELAY_SPREAD_S = 300e-9
SPEEDS_KMH = (3, 120)
SNRS_DB = np.arange(0, 31, 5)
N_SAMPLES = 2000
SEED = 0

COLORS = {"LS + interp.": "#2a78d6", "LMMSE": "#eb6834"}
LINESTYLES = {3: "-", 120: "--"}
MARKERS = {3: "o", 120: "s"}


def main():
    cfg = GridConfig()
    rng = np.random.default_rng(SEED)
    x_p = qpsk_pilots(cfg, rng)
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    rows = []
    for profile in PROFILES:
        for speed in SPEEDS_KMH:
            h = generate_channels(cfg, profile, DELAY_SPREAD_S, speed, N_SAMPLES, rng)
            r = channel_covariance(cfg, profile, DELAY_SPREAD_S, speed)
            for snr in SNRS_DB:
                h_ls_p = ls_pilots(received_pilots(h, cfg, x_p, snr, rng), x_p)
                rows.append((profile, speed, snr, "LS + interp.", nmse_db(ls_interp(h_ls_p, cfg), h)))
                w = lmmse_matrix(r, cfg, snr)
                rows.append((profile, speed, snr, "LMMSE", nmse_db(lmmse(h_ls_p, w, cfg), h)))

    with open(out_dir / "nmse_reference.csv", "w") as f:
        f.write("profile,speed_kmh,snr_db,estimator,nmse_db\n")
        for row in rows:
            f.write(",".join(map(str, row[:-1])) + f",{row[-1]:.3f}\n")

    fig, axes = plt.subplots(1, len(PROFILES), figsize=(13, 4.2), sharey=True)
    for ax, profile in zip(axes, PROFILES):
        for est, color in COLORS.items():
            for speed in SPEEDS_KMH:
                pts = [(s, v) for p, sp, s, e, v in rows if p == profile and sp == speed and e == est]
                s, v = zip(*pts)
                ax.plot(s, v, color=color, ls=LINESTYLES[speed], marker=MARKERS[speed],
                        lw=2, ms=6, label=f"{est}, {speed} km/h")
        ax.set_title(f"{profile}, DS = {DELAY_SPREAD_S * 1e9:.0f} ns", fontsize=11)
        ax.set_xlabel("SNR (dB)")
        ax.grid(True, color="#e0e0e0", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("NMSE (dB)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{cfg.n_sc}×{cfg.n_sym} ızgara, DMRS sembol {cfg.dmrs_symbols}, "
                 f"comb-{cfg.dmrs_comb}, fc = {cfg.fc_hz / 1e9:.1f} GHz", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "nmse_reference.png", dpi=150)

    print(f"{'profil':6} {'hız':>4} " + " ".join(f"{s:>6}" for s in SNRS_DB))
    for profile in PROFILES:
        for speed in SPEEDS_KMH:
            for est in COLORS:
                vals = [v for p, sp, s, e, v in rows if p == profile and sp == speed and e == est]
                print(f"{profile:6} {speed:>4} " + " ".join(f"{v:6.1f}" for v in vals) + f"  {est}")


if __name__ == "__main__":
    main()
