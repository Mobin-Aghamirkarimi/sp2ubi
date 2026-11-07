"""Per-beacon, per-head narrowband baseband of the vehicle's beacon after data-aided waveform removal.

After decoding, the receiver divides out the known beacon waveform, which leaves the time-varying channel times
the frequency offset: h(t) * exp(j 2 pi (eps_tx + eps_head) t) + noise. Because this residual occupies only a few
kHz, it is block-averaged to M samples per packet; block averaging of a narrowband signal is lossless, so the noise
variance per decimated sample is 1 / (SNR_raw * decimation).

Channel at head h (total power of LoS + diffuse = 1):
  LoS       sqrt(K/(K+1)) * exp(-j 2 pi r_h(t) / lambda)
  diffuse   sum_k g_hk * exp(j 2 pi (cos(phi_k) x(t) + sin(phi_k) y(t)) / lambda), E sum|g|^2 = 1/(K+1).
            The departure directions phi_k (hence the Doppler set) are common to both heads, the gains are
            independent per head: the same scatterers around the vehicle re-radiate towards both heads.
  specular  sqrt(rho_s * K/(K+1)) * exp(-j 2 pi (|p - p_r| + |p_r - head_h|) / lambda + j theta), a fixed point
            reflector (e.g. a facade across the road); unresolved in delay, i.e. worst case.
"""
from dataclasses import dataclass

import numpy as np

from .geometry import C, LAMBDA, Site, Trajectory, doppler, unit_to_heads


@dataclass(frozen=True)
class ChannelConfig:
    K_db: float = 20.0                  # Rician K-factor of the LoS tap
    scattering: str = "iso"             # "iso" | "side" (von Mises around +-90 deg)
    n_paths: int = 64
    specular_db: float | None = None    # specular reflector power relative to LoS (None = absent)
    packet_s: float = 1e-3              # beacon duration (C-V2X subframe)
    fs_raw: float = 10e6                # receiver sample rate before decimation
    M: int = 128                        # decimated samples per packet
    snr_ref_db: float = 40.0            # per-sample SNR at 100 m range
    cfo_random_hz: float = 590.0        # per-beacon randomised transmitter offset, uniform +-; cancels in FDOA
    head_offset_hz: tuple = (0.03, -0.03)  # residual head-to-head reference offset (0.01 ppb ~ 0.06 Hz)
    head_drift_hz_s: tuple = (0.0, 0.0)    # linear drift of each head's reference offset during the pass (Hz/s)
    K_db_head: tuple | None = None      # per-head K-factor (overrides K_db per head)
    block: tuple | None = None          # (head, start fraction, length fraction, attenuation dB): LoS of one head
                                        # attenuated for a contiguous part of the pass (blockage by another vehicle)
    # wideband part (time of arrival from the DMRS channel estimate; LTE-V2X 10 MHz numerology)
    n_sc: int = 600                     # occupied subcarriers
    scs_hz: float = 15e3
    n_dmrs: int = 4                     # DMRS symbols per subframe averaged in the channel estimate
    diffuse_delay_ns: float = 30.0      # mean excess delay of the diffuse (vehicle-local) paths, exponential
    toa_oversample: int = 16


def _departure_angles(rng, cfg: ChannelConfig):
    if cfg.scattering == "iso":
        return rng.uniform(0, 2 * np.pi, cfg.n_paths)
    if cfg.scattering == "side":
        side = rng.choice([-np.pi / 2, np.pi / 2], cfg.n_paths)
        return side + rng.vonmises(0.0, 4.0, cfg.n_paths)
    raise ValueError(cfg.scattering)


def draw_reflector(rng, site: Site):
    """A specular point reflector across the road (facade, parked truck) and its reflection phase."""
    return np.array([rng.uniform(0, site.D), -15.0, rng.uniform(0, 4.0)]), rng.uniform(0, 2 * np.pi)


def _toa_peak(H, cfg: ChannelConfig):
    """Delay (s) of the strongest peak of the oversampled channel impulse response, per row of H (n_b, n_sc)."""
    L = cfg.n_sc * cfg.toa_oversample
    h = np.abs(np.fft.ifft(H, L, axis=-1)) ** 2
    k = np.argmax(h, axis=-1); idx = np.arange(h.shape[0])
    a, b, c = h[idx, (k - 1) % L], h[idx, k], h[idx, (k + 1) % L]
    d = 0.5 * (a - c) / (a - 2 * b + c)
    kk = (k + d + L / 2) % L - L / 2                           # signed delay bin
    return kk / (L * cfg.scs_hz)


def simulate_pass(rng, site: Site, traj: Trajectory, cfg: ChannelConfig, beacon_hz: float = 10.0, wideband=False,
                  reflector=None, doppler_scale=1.0):
    """Return (t_b, y, f_true) with y of shape (2, n_b, M) and f_true the true per-head LoS Doppler (2, n_b).
    With wideband=True also return (toa_hat, toa_true), shape (2, n_b): times of arrival at both heads relative to a
    common, unknown per-beacon transmit-time offset (it cancels in TDOA = toa_A - toa_B).
    reflector: optional (p_r, theta_r) of a site-fixed specular reflector; drawn per pass when None.
    doppler_scale: frequency-only manipulation. Every path's Doppler is multiplied by doppler_scale while the
    emitter's position follows the true trajectory -- the signature of a moving-phase-centre (commutated) array or a
    frequency-shifting surface on the vehicle, which mimics a different speed in the frequency domain only. A float
    (constant) or ("ramp", beta_max): scale rises linearly from 1 to 1 + beta_max over the pass.
    reflector may carry a third element v_r (m/s along +x): a moving specular reflector (e.g. a truck in the adjacent
    lane); its Doppler is then the numerical derivative of the full two-hop path length."""
    t_b = np.arange(traj.t[0] + 0.05, traj.t[-1] - 0.05, 1.0 / beacon_hz)
    t_b = t_b + rng.uniform(0, 1.0 / beacon_hz)            # random phase of the beacon schedule
    t_b = t_b[t_b < traj.t[-1] - cfg.packet_s]
    tau = (np.arange(cfg.M) + 0.5) / cfg.M * cfg.packet_s - cfg.packet_s / 2   # packet-centred time
    x, y, vx, vy = traj.at(t_b)
    frontback = 0.0
    if isinstance(doppler_scale, tuple) and doppler_scale[0] == "frontback":
        # two-antenna relay: forward/backward antennas add opposite offsets beta*|v|/lambda*sign(cos);
        # needs no knowledge of the head positions
        frontback, ds = doppler_scale[1], np.ones(len(t_b))
    elif isinstance(doppler_scale, tuple):
        ds = 1.0 + doppler_scale[1] * (t_b - t_b[0]) / max(t_b[-1] - t_b[0], 1e-9)
    else:
        ds = np.full(len(t_b), float(doppler_scale))
    fD = doppler(site, x, y, vx, vy) * ds                     # (2, n_b) at packet centres
    if frontback:
        u_h, _ = unit_to_heads(site, x, y)
        fD = fD + frontback * np.abs(vx)[None, :] / LAMBDA * np.sign(u_h[:, :, 0])
    _, r = unit_to_heads(site, x, y)                          # (2, n_b)
    K = 10 ** (cfg.K_db / 10)
    a_los = np.sqrt(K / (K + 1))
    phi = _departure_angles(rng, cfg)
    f_k = ds[:, None] * (np.outer(vx, np.cos(phi)) + np.outer(vy, np.sin(phi))) / LAMBDA          # (n_b, P)
    if frontback:
        f_k = f_k + frontback * np.abs(vx)[:, None] / LAMBDA * np.sign(np.cos(phi))[None, :]
    ph_k = 2 * np.pi * (np.outer(x, np.cos(phi)) + np.outer(y, np.sin(phi))) / LAMBDA
    eps_tx = rng.uniform(-cfg.cfo_random_hz, cfg.cfo_random_hz, len(t_b))
    dec = cfg.fs_raw * cfg.packet_s / cfg.M
    out = np.empty((2, len(t_b), cfg.M), complex)
    toa_hat = np.empty((2, len(t_b))); toa_true = r / C
    fk = (np.arange(cfg.n_sc) - cfg.n_sc / 2) * cfg.scs_hz
    t_off = rng.uniform(-1e-6, 1e-6, len(t_b))                 # unknown common transmit-time offset per beacon
    if cfg.specular_db is not None:
        refl = reflector if reflector is not None else draw_reflector(rng, site)
        p_r, theta_r = refl[0], refl[1]
        v_r = refl[2] if len(refl) > 2 else 0.0
    for h in range(2):
        if cfg.K_db_head is not None:                          # per-head K (None: identical to the single-K model)
            K = 10 ** (cfg.K_db_head[h] / 10)
            a_los = np.sqrt(K / (K + 1))
        g = (rng.standard_normal(cfg.n_paths) + 1j * rng.standard_normal(cfg.n_paths)) / np.sqrt(2 * cfg.n_paths * (K + 1))
        los = a_los * np.exp(-2j * np.pi * r[h] / LAMBDA)[:, None] * np.exp(2j * np.pi * fD[h][:, None] * tau[None, :])
        if cfg.block is not None and cfg.block[0] == h:
            n_b = len(t_b); i0 = int(cfg.block[1] * n_b); i1 = min(n_b, i0 + max(1, int(cfg.block[2] * n_b)))
            los[i0:i1] *= 10 ** (-cfg.block[3] / 20)
        diff = np.einsum("p,bpm->bm", g, np.exp(1j * (ph_k[:, :, None] + 2 * np.pi * f_k[:, :, None] * tau[None, None, :])))
        sig = los + diff
        if cfg.specular_db is not None:
            head = site.heads()[h]
            def path_len(tq):
                xq, yq, _, _ = traj.at(tq)
                pr = p_r[None, :] + np.outer(tq - traj.t[0], [v_r, 0.0, 0.0])
                pos = np.stack([xq, yq, np.zeros_like(xq)], axis=-1)
                return np.linalg.norm(pos - pr, axis=-1) + np.linalg.norm(pr - head, axis=-1)
            L = path_len(t_b)
            f_r = ds * -(path_len(t_b + 1e-3) - path_len(t_b - 1e-3)) / (2e-3 * LAMBDA)
            amp = np.sqrt(10 ** (cfg.specular_db / 10)) * a_los
            sig = sig + amp * np.exp(1j * (theta_r - 2 * np.pi * L / LAMBDA))[:, None] * np.exp(2j * np.pi * f_r[:, None] * tau[None, :])
        snr = 10 ** ((cfg.snr_ref_db - 20 * np.log10(np.maximum(r[h], 1.0) / 100.0)) / 10)
        noise_std = np.sqrt(1.0 / (snr * dec) / 2)[:, None]
        noise = noise_std * (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape))
        head_off = cfg.head_offset_hz[h] + cfg.head_drift_hz_s[h] * (t_b - t_b[0])
        cfo = np.exp(2j * np.pi * (eps_tx + head_off)[:, None] * tau[None, :])
        out[h] = sig * cfo + noise
        if wideband:
            # channel frequency response at packet centre; delays relative to (r_h/c + t_off), the common reference
            ex = rng.exponential(cfg.diffuse_delay_ns * 1e-9, cfg.n_paths)
            amp_los = a_los * np.exp(-2j * np.pi * r[h] / LAMBDA)                  # (n_b)
            amp_d = g[None, :] * np.exp(1j * ph_k)                                  # (n_b, P)
            H = amp_los[:, None] * np.ones((1, cfg.n_sc)) + amp_d @ np.exp(-2j * np.pi * np.outer(ex, fk))
            if cfg.specular_db is not None:
                ex_s = (L - r[h]) / C
                H = H + (amp * np.exp(1j * (theta_r - 2 * np.pi * L / LAMBDA)))[:, None] * np.exp(-2j * np.pi * np.outer(ex_s, fk))
            ns = np.sqrt(1.0 / (snr * cfg.n_dmrs) / 2)[:, None]
            H = H + ns * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
            toa_hat[h] = r[h] / C + t_off + _toa_peak(H, cfg)
    if wideband:
        return t_b, out, fD, toa_hat, toa_true + t_off
    return t_b, out, fD
