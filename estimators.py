"""Klasik referans kestiriciler: LS + doğrusal interpolasyon ve 2B LMMSE."""
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import j0

from ofdm_channel import GridConfig, tdl_taps


def ls_pilots(y_p: np.ndarray, x_p: np.ndarray) -> np.ndarray:
    return y_p / x_p


def ls_interp(h_ls_p: np.ndarray, cfg: GridConfig) -> np.ndarray:
    """Pilot LS kestirimlerini önce frekansta, sonra zamanda doğrusal olarak tam ızgaraya yayar.

    Kenarlarda doğrusal ekstrapolasyon yapılır. Çıktı: (N, n_sc, n_sym).
    """
    n = h_ls_p.shape[0]
    k_p = np.arange(0, cfg.n_sc, cfg.dmrs_comb)
    l_p = np.array(cfg.dmrs_symbols)
    h_p = h_ls_p.reshape(n, k_p.size, l_p.size)    # pilot_mask sırası: önce k, sonra l

    f_interp = interp1d(k_p, h_p, axis=1, kind="linear", fill_value="extrapolate")
    h_f = f_interp(np.arange(cfg.n_sc))            # (N, K, n_dmrs)
    t_interp = interp1d(l_p, h_f, axis=2, kind="linear", fill_value="extrapolate")
    return t_interp(np.arange(cfg.n_sym))


def channel_covariance(cfg: GridConfig, profile: str, delay_spread_s: float,
                       speed_kmh: float) -> np.ndarray:
    """vec(H) için ayrışabilir kovaryans R = R_f ⊗ R_t, (K·L × K·L).

    R_f: güç gecikme profilinden; R_t: Jakes modeli, J0(2π f_D Δt).
    """
    delays, powers = tdl_taps(profile, delay_spread_s)
    dk = np.subtract.outer(np.arange(cfg.n_sc), np.arange(cfg.n_sc))
    r_f = (powers[:, None, None] * np.exp(-2j * np.pi * cfg.scs_hz * delays[:, None, None] * dk)).sum(0)
    dl = np.subtract.outer(np.arange(cfg.n_sym), np.arange(cfg.n_sym))
    r_t = j0(2 * np.pi * cfg.doppler_hz(speed_kmh) * dl * cfg.t_sym)
    return np.kron(r_f, r_t)


def lmmse_matrix(r: np.ndarray, cfg: GridConfig, snr_db: float) -> np.ndarray:
    """W = R_hp (R_pp + σ² I)^-1; h_hat = h_ls_p @ W.T. Kanal istatistiklerinin
    tam bilindiğini varsayar (ideal LMMSE, üst sınır niteliğinde referans)."""
    p = np.flatnonzero(cfg.pilot_mask().ravel())
    sigma2 = 10 ** (-snr_db / 10)
    r_pp = r[np.ix_(p, p)] + sigma2 * np.eye(p.size)
    return np.linalg.solve(r_pp.T, r[:, p].T).T     # (K·L, n_pilot)


def lmmse(h_ls_p: np.ndarray, w: np.ndarray, cfg: GridConfig) -> np.ndarray:
    return (h_ls_p @ w.T).reshape(-1, cfg.n_sc, cfg.n_sym)


def nmse_db(h_hat: np.ndarray, h: np.ndarray) -> float:
    return 10 * np.log10(np.sum(np.abs(h_hat - h) ** 2) / np.sum(np.abs(h) ** 2))
