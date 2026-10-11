"""OFDM ızgarası, 3GPP TDL kanal modeli ve DMRS pilot deseni (yalnızca NumPy).

Kanal doğrudan frekans alanında üretilir:
    H[k, l] = sum_p sqrt(P_p) * g_p(t_l) * exp(-j 2π k Δf τ_p)
Varsayımlar: CP en büyük gecikmeden uzun (ISI yok), sembol içi Doppler ihmal (ICI yok),
tek anten (SISO), her tap Rayleigh sönümlemeli (TDL-A/B/C NLOS profilleri).
"""
from dataclasses import dataclass

import numpy as np

# 3GPP TR 38.901 Tablo 7.7.2-1/2/3: (normalize gecikme, güç [dB])
TDL_PROFILES = {
    "TDL-A": np.array([
        [0.0000, -13.4], [0.3819, 0.0], [0.4025, -2.2], [0.5868, -4.0],
        [0.4610, -6.0], [0.5375, -8.2], [0.6708, -9.9], [0.5750, -10.5],
        [0.7618, -7.5], [1.5375, -15.9], [1.8978, -6.6], [2.2242, -16.7],
        [2.1718, -12.4], [2.4942, -15.2], [2.5119, -10.8], [3.0582, -11.3],
        [4.0810, -12.7], [4.4579, -16.2], [4.5695, -18.3], [4.7966, -18.9],
        [5.0066, -16.6], [5.3043, -19.9], [9.6586, -29.7],
    ]),
    "TDL-B": np.array([
        [0.0000, 0.0], [0.1072, -2.2], [0.2155, -4.0], [0.2095, -3.2],
        [0.2870, -9.8], [0.2986, -1.2], [0.3752, -3.4], [0.5055, -5.2],
        [0.3681, -7.6], [0.3697, -3.0], [0.5700, -8.9], [0.5283, -9.0],
        [1.1021, -4.8], [1.2756, -5.7], [1.5474, -7.5], [1.7842, -1.9],
        [2.0169, -7.6], [2.8294, -12.2], [3.0219, -9.8], [3.6187, -11.4],
        [4.1067, -14.9], [4.2790, -9.2], [4.7834, -11.3],
    ]),
    "TDL-C": np.array([
        [0.0000, -4.4], [0.2099, -1.2], [0.2219, -3.5], [0.2329, -5.2],
        [0.2176, -2.5], [0.6366, 0.0], [0.6448, -2.2], [0.6560, -3.9],
        [0.6584, -7.4], [0.7935, -7.1], [0.8213, -10.7], [0.9336, -11.1],
        [1.2285, -5.1], [1.3083, -6.8], [2.1704, -8.7], [2.7105, -13.2],
        [4.2589, -13.9], [4.6003, -13.9], [5.4902, -15.8], [5.6077, -17.1],
        [6.3065, -16.0], [6.6374, -15.7], [7.0427, -21.6], [8.6523, -22.8],
    ]),
}

C_LIGHT = 299_792_458.0


@dataclass(frozen=True)
class GridConfig:
    n_sc: int = 72                  # 6 RB × 12 alt taşıyıcı
    n_sym: int = 14                 # bir slot, normal CP
    scs_hz: float = 15e3
    fc_hz: float = 3.5e9
    dmrs_symbols: tuple = (2, 11)   # front-loaded + 1 ek DMRS
    dmrs_comb: int = 2              # Type-1 benzeri: her 2 alt taşıyıcıda bir

    @property
    def t_sym(self) -> float:
        # 14 sembol = 1 ms (15 kHz); CP dahil ortalama sembol süresi
        return 1e-3 * (15e3 / self.scs_hz) / 14

    def pilot_mask(self) -> np.ndarray:
        """(n_sc, n_sym) boolean pilot maskesi."""
        mask = np.zeros((self.n_sc, self.n_sym), dtype=bool)
        for l in self.dmrs_symbols:
            mask[::self.dmrs_comb, l] = True
        return mask

    def doppler_hz(self, speed_kmh: float) -> float:
        return speed_kmh / 3.6 * self.fc_hz / C_LIGHT


def tdl_taps(profile: str, delay_spread_s: float):
    """Gecikmeler [s] ve toplamı 1'e normalize edilmiş doğrusal güçler."""
    table = TDL_PROFILES[profile]
    delays = table[:, 0] * delay_spread_s
    powers = 10 ** (table[:, 1] / 10)
    return delays, powers / powers.sum()


def generate_channels(cfg: GridConfig, profile: str, delay_spread_s: float,
                      speed_kmh: float, n_samples: int, rng: np.random.Generator,
                      n_sinusoids: int = 16) -> np.ndarray:
    """(n_samples, n_sc, n_sym) karmaşık kanal ızgaraları, E|H|^2 = 1.

    Her tap için Jakes Doppler spektrumu, rastgele geliş açılı sinüs toplamı ile üretilir.
    """
    delays, powers = tdl_taps(profile, delay_spread_s)
    n_taps = delays.size
    f_d = cfg.doppler_hz(speed_kmh)
    t = np.arange(cfg.n_sym) * cfg.t_sym

    alpha = rng.uniform(0, 2 * np.pi, (n_samples, n_taps, n_sinusoids, 1))
    phi = rng.uniform(0, 2 * np.pi, (n_samples, n_taps, n_sinusoids, 1))
    g = np.exp(1j * (2 * np.pi * f_d * np.cos(alpha) * t + phi)).sum(axis=2)
    g /= np.sqrt(n_sinusoids)                                   # (N, P, L)

    k = np.arange(cfg.n_sc)
    steer = np.sqrt(powers)[:, None] * np.exp(-2j * np.pi * cfg.scs_hz * np.outer(delays, k))
    return np.einsum("npl,pk->nkl", g, steer)                   # (N, K, L)


def qpsk_pilots(cfg: GridConfig, rng: np.random.Generator) -> np.ndarray:
    """Pilot konumlarında birim güçlü QPSK semboller (sabit desen, tüm örneklerde aynı)."""
    n_p = int(cfg.pilot_mask().sum())
    bits = rng.integers(0, 2, (n_p, 2))
    return ((1 - 2 * bits[:, 0]) + 1j * (1 - 2 * bits[:, 1])) / np.sqrt(2)


def received_pilots(h: np.ndarray, cfg: GridConfig, x_p: np.ndarray, snr_db,
                    rng: np.random.Generator) -> np.ndarray:
    """Pilot RE'lerinde Y = H·X + N; (N, n_pilot). SNR, RE başına E|H|^2/σ².

    snr_db skaler ya da örnek başına (N,) dizi olabilir.
    """
    h_p = h[:, cfg.pilot_mask()]
    sigma2 = 10 ** (-np.asarray(snr_db, dtype=float) / 10)
    if sigma2.ndim == 1:
        sigma2 = sigma2[:, None]
    noise = np.sqrt(sigma2 / 2) * (rng.standard_normal(h_p.shape) + 1j * rng.standard_normal(h_p.shape))
    return h_p * x_p + noise
