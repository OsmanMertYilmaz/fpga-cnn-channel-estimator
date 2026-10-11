"""Aşama 2 değerlendirmesi: CNN modellerini LS + interpolasyon ve ideal LMMSE ile aynı senaryolarda karşılaştırır.

NMSE, üç TDL profili üzerinden ortalanır (profiller bu ayarda neredeyse aynı sonucu veriyor).
Çalıştırma:  .venv/bin/python evaluate.py --models cnn_fp32 cnn_snr --out nmse_compare
Çıktılar:    results/<out>.png, results/<out>.csv
"""
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataset import fixed_pilots, from_patches, model_inputs, to_patches
from estimators import channel_covariance, lmmse, lmmse_matrix, ls_interp, ls_pilots
from ofdm_channel import GridConfig, generate_channels, received_pilots

PROFILES = ("TDL-A", "TDL-B", "TDL-C")
DELAY_SPREAD_S = 300e-9
SPEEDS_KMH = (3, 120)
SNRS_DB = np.arange(0, 31, 5)
N_SAMPLES = 1000
SEED = 7                           # eğitim/doğrulama tohumlarından farklı

# Sabit sıra: LS mavi, LMMSE turuncu, modeller sırayla aqua, sarı, magenta, yeşil
REF_COLORS = {"LS + interp.": "#2a78d6", "LMMSE (ideal)": "#eb6834"}
MODEL_COLORS = ("#1baf7a", "#eda100", "#e87ba4", "#008300")
MARKERS = ("o", "s", "^", "D", "v", "P")


def predict_grid(model, h_ls: np.ndarray, snr_db: float, n_sc: int) -> np.ndarray:
    _, patch_sc, _, n_ch = model.input_shape
    x = to_patches(h_ls)
    x = model_inputs(x, np.full(x.shape[0], snr_db), patch_sc, n_ch == 3)
    return from_patches(model.predict(x, batch_size=2048, verbose=0), n_sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["cnn_fp32"])
    ap.add_argument("--out", default="nmse_cnn")
    args = ap.parse_args()

    root = Path(__file__).parent
    cfg = GridConfig()
    x_p = fixed_pilots(cfg)
    rng = np.random.default_rng(SEED)
    models = {m: tf.keras.models.load_model(root / f"models/{m}.keras", compile=False) for m in args.models}
    names = list(REF_COLORS) + list(models)

    # err[est][speed][snr] = (hata enerjisi, kanal enerjisi) profiller üzerinden toplanır
    err = {e: {s: np.zeros((len(SNRS_DB), 2)) for s in SPEEDS_KMH} for e in names}
    for profile in PROFILES:
        for speed in SPEEDS_KMH:
            h = generate_channels(cfg, profile, DELAY_SPREAD_S, speed, N_SAMPLES, rng)
            r = channel_covariance(cfg, profile, DELAY_SPREAD_S, speed)
            for i, snr in enumerate(SNRS_DB):
                h_ls_p = ls_pilots(received_pilots(h, cfg, x_p, snr, rng), x_p)
                h_ls = ls_interp(h_ls_p, cfg)
                est = {"LS + interp.": h_ls, "LMMSE (ideal)": lmmse(h_ls_p, lmmse_matrix(r, cfg, snr), cfg)}
                for m, model in models.items():
                    est[m] = predict_grid(model, h_ls, snr, cfg.n_sc)
                for e, h_hat in est.items():
                    err[e][speed][i] += (np.sum(np.abs(h_hat - h) ** 2), np.sum(np.abs(h) ** 2))
    nmse = {e: {s: 10 * np.log10(v[:, 0] / v[:, 1]) for s, v in d.items()} for e, d in err.items()}

    with open(root / f"results/{args.out}.csv", "w") as f:
        f.write("estimator,speed_kmh," + ",".join(f"snr{s}" for s in SNRS_DB) + "\n")
        for e in names:
            for s in SPEEDS_KMH:
                f.write(f"{e},{s}," + ",".join(f"{v:.3f}" for v in nmse[e][s]) + "\n")

    colors = dict(REF_COLORS, **dict(zip(models, MODEL_COLORS)))
    fig, axes = plt.subplots(1, len(SPEEDS_KMH), figsize=(11, 4.4), sharey=True)
    for ax, speed in zip(axes, SPEEDS_KMH):
        for j, e in enumerate(names):
            label = e if e in REF_COLORS else f"{e} ({models[e].count_params()} p.)"
            ax.plot(SNRS_DB, nmse[e][speed], color=colors[e], marker=MARKERS[j % len(MARKERS)],
                    lw=2, ms=6, label=label)
        ax.set_title(f"{speed} km/h", fontsize=11)
        ax.set_xlabel("SNR (dB)")
        ax.grid(True, color="#e0e0e0", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("NMSE (dB)")
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle(f"TDL-A/B/C ortalaması, DS = {DELAY_SPREAD_S * 1e9:.0f} ns, {cfg.n_sc}×{cfg.n_sym} ızgara",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(root / f"results/{args.out}.png", dpi=150)

    print(f"{'kestirici':24} {'hız':>4} " + " ".join(f"{s:>6}" for s in SNRS_DB))
    for s in SPEEDS_KMH:
        for e in names:
            print(f"{e:24} {s:>4} " + " ".join(f"{v:6.1f}" for v in nmse[e][s]))


if __name__ == "__main__":
    main()
