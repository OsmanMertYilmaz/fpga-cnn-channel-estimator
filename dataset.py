"""Aşama 1: eğitim/doğrulama veri seti.

Girdi: LS + doğrusal interpolasyon kestirimi, hedef: gerçek kanal. Her 72×14 ızgara
12×14'lük 6 yamaya bölünür (gerçek/sanal → 2 kanal), CNN yalnızca gürültü giderir.

Çalıştırma:  .venv/bin/python dataset.py
Çıktı:       data/train.npz, data/val.npz
"""
from pathlib import Path

import numpy as np

from estimators import ls_interp, ls_pilots
from ofdm_channel import GridConfig, generate_channels, qpsk_pilots, received_pilots

PATCH_SC = 12
PROFILES = ("TDL-A", "TDL-B", "TDL-C")
DS_RANGE_S = (50e-9, 300e-9)
SPEED_RANGE_KMH = (0.0, 120.0)
SNR_RANGE_DB = (0.0, 30.0)
CHUNK = 250                       # aynı (profil, DS, hız) ile üretilen ızgara sayısı
SPLITS = {"train": 16000, "val": 2000}
PILOT_SEED = 1234                 # sabit DMRS deseni: tüm betikler aynı pilotları kullanır


def fixed_pilots(cfg: GridConfig) -> np.ndarray:
    return qpsk_pilots(cfg, np.random.default_rng(PILOT_SEED))


def to_patches(h: np.ndarray) -> np.ndarray:
    """(N, K, L) karmaşık → (N·K/12, 12, L, 2) gerçek/sanal yamalar."""
    n, k, l = h.shape
    x = np.stack([h.real, h.imag], axis=-1).astype(np.float32)
    return x.reshape(n * (k // PATCH_SC), PATCH_SC, l, 2)


def from_patches(x: np.ndarray, n_sc: int) -> np.ndarray:
    """to_patches'in tersi: (N·K/12, 12, L, 2) → (N, K, L) karmaşık."""
    x = x.reshape(-1, n_sc, x.shape[2], 2)
    return x[..., 0] + 1j * x[..., 1]


SNR_SCALE_DB = 30.0               # SNR girdi düzlemi: snr_db / 30 → yaklaşık [0, 1]


def model_inputs(x: np.ndarray, snr_db: np.ndarray, patch_sc: int, snr_input: bool) -> np.ndarray:
    """12'lik yamaları patch_sc'lik yamalara birleştirir; istenirse 3. kanal olarak sabit SNR düzlemi ekler.

    x: (M, 12, L, 2) ızgara sırasıyla ardışık yamalar, snr_db: (M,) yama başına SNR.
    """
    group = patch_sc // PATCH_SC
    x = x.reshape(-1, patch_sc, x.shape[2], 2)
    if not snr_input:
        return x
    snr = np.asarray(snr_db, dtype=np.float32).reshape(-1, group)[:, 0] / SNR_SCALE_DB
    plane = np.broadcast_to(snr[:, None, None, None], x.shape[:3] + (1,))
    return np.concatenate([x, plane], axis=-1)


def make_split(cfg: GridConfig, x_p: np.ndarray, n_grids: int, rng: np.random.Generator):
    inputs, targets, snrs = [], [], []
    for _ in range(n_grids // CHUNK):
        profile = rng.choice(PROFILES)
        ds = rng.uniform(*DS_RANGE_S)
        speed = rng.uniform(*SPEED_RANGE_KMH)
        h = generate_channels(cfg, profile, ds, speed, CHUNK, rng)
        snr = rng.uniform(*SNR_RANGE_DB, CHUNK)
        h_ls = ls_interp(ls_pilots(received_pilots(h, cfg, x_p, snr, rng), x_p), cfg)
        inputs.append(to_patches(h_ls))
        targets.append(to_patches(h))
        snrs.append(np.repeat(snr, cfg.n_sc // PATCH_SC))
    return np.concatenate(inputs), np.concatenate(targets), np.concatenate(snrs).astype(np.float32)


def main():
    cfg = GridConfig()
    x_p = fixed_pilots(cfg)
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    for i, (name, n_grids) in enumerate(SPLITS.items()):
        x, y, snr = make_split(cfg, x_p, n_grids, np.random.default_rng(100 + i))
        np.savez(out_dir / f"{name}.npz", x=x, y=y, snr=snr)
        print(f"{name}: {x.shape[0]} yama {x.shape[1:]}, "
              f"{(x.nbytes + y.nbytes) / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
