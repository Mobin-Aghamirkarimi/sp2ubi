"""The speed-check simulation interface: turns generator output into the per-beacon, per-head narrowband baseband that the
site-accuracy pipeline consumes (fdoa_from_samples -> fit_pass_offset -> statistics), and wraps the in-house channel
(unchanged) so that both arms get identical post-processing with common random
numbers: transmitter offset per beacon (common to both heads), head-to-head offset, thermal noise, receiver impairments (separate module below).
Conventions: coefficients carry the carrier phase exp(-j 2 pi d / lambda); the LoS Doppler is positive when the path
shortens (as in fdoa.channel); packet-centred time tau in [-0.5, 0.5) ms; M samples per beacon."""
import dataclasses
import json
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline

ER = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(ER, ".."), os.path.join(ER, "..", "inhouse")]
from fdoa.channel import ChannelConfig, simulate_pass  # noqa: E402
from fdoa.geometry import C, LAMBDA  # noqa: E402

SCS = 15e3


def tau_grid(M=128, packet_s=1e-3):
    return (np.arange(M) + 0.5) / M * packet_s - packet_s / 2


def band_weights(delay_rel, prb):
    """Average of exp(-j 2 pi f_k tau) over the occupied subcarriers (12 * prb, centred): the beacon's narrowband
    view of a path with relative delay tau (s). delay_rel: array (...)."""
    n = 12 * int(prb)
    # closed form of mean_k exp(-j 2 pi (k - (n-1)/2) SCS tau), k = 0..n-1 (Dirichlet kernel; real because the grid is
    # centred): no [..., n] array, so the stage-1 memory does not grow with the band; equal to
    # the explicit mean within 1e-12 (tests/test_band_weights.py)
    x = np.pi * SCS * np.asarray(delay_rel, float)
    s = np.sin(x)
    small = np.abs(s) < 1e-12
    k = np.round(x / np.pi)                              # at tau = k / SCS the limit is (-1)^(k (n-1))
    w = np.where(small, np.where((k * (n - 1)) % 2 == 0, 1.0, -1.0), np.sin(n * x) / (n * np.where(small, 1.0, s)))
    return w.astype(complex)


def band_avg_q(coeff, delay, Q, prb, packet_s=1e-3):
    """coeff, delay: [paths, n_beacon * Q] at the Q snapshot instants per beacon (beacon-major, centred in the
    packet). Returns h [n_beacon, Q]: band average over the occupied subcarriers of the sum over paths, delays relative
    to path 0 (the LoS, i.e. timing synchronised to the first arrival). This is the stored canonical channel."""
    P, S = coeff.shape
    nb = S // Q
    c = coeff.reshape(P, nb, Q)
    d = delay.reshape(P, nb, Q)
    w = band_weights(d - d[0:1], prb)                  # [P, nb, Q]
    return np.sum(c * w, axis=0)


def interp_q(h, M=128, packet_s=1e-3, time_scale=1.0):
    """Complex cubic interpolation of h [n_beacon, Q] (Q centred snapshots per beacon) to M samples per beacon.
    time_scale = 1 + beta evaluates h((1 + beta) tau): with each path's phase linear in tau inside the 1 ms beacon this is
    the channel with every path's Doppler (LoS, clusters, scatterer-mobility term) scaled by (1 + beta) and geometry
    and delays unchanged (frequency-only manipulation; doppler_scale in fdoa.channel)."""
    Q = h.shape[1]
    tq = (np.arange(Q) + 0.5) / Q * packet_s - packet_s / 2
    cs = CubicSpline(tq, h, axis=1, extrapolate=True)
    return cs(tau_grid(M, packet_s) * time_scale)


def narrowband_from_paths(coeff, delay, Q, prb, M=128, packet_s=1e-3):
    """band_avg_q followed by interp_q: [n_beacon, M]."""
    return interp_q(band_avg_q(coeff, delay, Q, prb, packet_s), M, packet_s)


def los_paths(site_heads, traj, t_snap):
    """Analytic LoS path per head at the snapshot instants (C8 null test): coeff exp(-j 2 pi d / lambda), delay d / c.
    site_heads: [2, 3] in the in-house frame (vehicle antenna at z = 0)."""
    x, y, _, _ = traj.at(t_snap)
    pos = np.stack([x, y, np.zeros_like(x)], axis=-1)
    out = []
    for h in range(2):
        d = np.linalg.norm(site_heads[h][None, :] - pos, axis=-1)
        out.append((np.exp(-2j * np.pi * d / LAMBDA)[None, :], (d / C)[None, :]))
    return out


def inhouse_clean(rng_schedule, site, traj, cfg: ChannelConfig, **kw):
    """In-house channel (unchanged simulate_pass) without transmitter offset, head offset and noise; the schedule rng
    must be a fresh copy of the one used for the generator arm so that both arms get the same beacon times."""
    clean = dataclasses.replace(cfg, cfo_random_hz=0.0, head_offset_hz=(0.0, 0.0), head_drift_hz_s=(0.0, 0.0),
                                snr_ref_db=400.0)
    t_b, y, fD = simulate_pass(rng_schedule, site, traj, clean, **kw)
    return t_b, y, fD


_C = json.load(open(os.environ.get("SPEEDCHECK_CONSTANTS", os.path.join(ER, "config", "study1_constants.json"))))
LINK_BUDGET = _C["link_budget"]


def beacon_gain(ls, Q):
    """Per-beacon large-scale power gain (linear) from the per-snapshot sum of ray powers [n_beacon * Q]."""
    return np.asarray(ls, float).reshape(-1, Q).mean(axis=1)


def snr_e1_db(r, snr_ref_db=40.0):
    """the site-accuracy simulation mapping: raw-sample SNR (dB) at fs_raw, 40 dB at 100 m, free-space slope (fdoa.channel)."""
    return snr_ref_db - 20 * np.log10(np.maximum(np.asarray(r, float), 1.0) / 100.0)


def snr_link_budget_db(gain, tx_dbm=LINK_BUDGET["tx_dbm"], nf_db=LINK_BUDGET["ue_nf_db"],
                       fs_raw=LINK_BUDGET["noise_bandwidth_hz"]):
    """TR 37.885 link budget: raw-sample SNR (dB) at fs_raw = P_rx / (k T NF fs_raw), P_rx = P_tx * gain (the generator's
    own large-scale gain incl. both antenna patterns). Equal-energy reading: the beacon energy P_rx * T is collected over
    the M samples whatever the PRB allocation, which is how the in-house site-accuracy mapping is defined (noise at fs_raw, gain fs_raw T / M
    from the decimation)."""
    usable = 10 * np.log10(LINK_BUDGET.get("usable_energy_fraction", 1.0))       # 12/14 symbols
    return tx_dbm + usable + 10 * np.log10(np.asarray(gain, float)) - (-174.0 + nf_db + 10 * np.log10(fs_raw))


def finish(rng, sig, snr_db, gain=None, cfo_random_hz=300.0, head_offset_hz=(0.0, 0.0), head_drift_hz_s=(0.0, 0.0),
           t_b=None, fs_raw=10e6, packet_s=1e-3):
    """Common post-processing of both C1 arms. sig: [2, n_b, M] clean baseband; snr_db: [2, n_b] raw-sample SNR per
    beacon (snr_e1_db or snr_link_budget_db; both arms of a pair get the same array); gain: [2, n_b] large-scale gain
    of a generator arm, divided out per beacon so that the channel has unit expected power as in the site-accuracy simulation (None for the
    in-house arm, whose channel already has unit expected power). Draw order is fixed: transmitter offsets, then noise
    of head A, then head B."""
    _, n_b, M = sig.shape
    tau = tau_grid(M, packet_s)
    eps_tx = rng.uniform(-cfo_random_hz, cfo_random_hz, n_b)
    t_rel = (t_b - t_b[0]) if t_b is not None else np.zeros(n_b)
    dec = fs_raw * packet_s / M
    out = np.empty_like(sig)
    for h in range(2):
        s = sig[h] if gain is None else sig[h] / np.sqrt(np.asarray(gain[h], float))[:, None]
        snr = 10 ** (np.asarray(snr_db[h], float) / 10)
        noise_std = np.sqrt(1.0 / (snr * dec) / 2)[:, None]
        noise = noise_std * (rng.standard_normal(s.shape) + 1j * rng.standard_normal(s.shape))
        off = head_offset_hz[h] + head_drift_hz_s[h] * t_rel
        out[h] = s * np.exp(2j * np.pi * (eps_tx + off)[:, None] * tau[None, :]) + noise
    return out


# ------------------------------------------------------------------------------------- receiver impairments
PN_PROFILE = {1.0: -19.6, 10.0: -54.6, 100.0: -76.6, 1e3: -86.6, 1e4: -89.6, 1e5: -94.6}   # dBc/Hz at 5.9 GHz


def phase_noise(rng, n_beacon, M=128, packet_s=1e-3, profile=None, extra_db=0.0, block=16):
    """Independent LO phase-noise realisations per beacon, [n_beacon, M] (rad): Gaussian process with one-sided phase
    PSD S_phi(f) = 2 * 10^(L(f)/10) rad^2/Hz, L(f) log-log interpolated from the profile (+ extra_db), synthesised on a
    block of `block` beacon durations at fs = M / packet_s and cut to the central beacon (components below
    1 / (block * packet_s) are not represented; slower drift is covered by the head-offset drift)."""
    prof = PN_PROFILE if profile is None else profile
    fs = M / packet_s
    N = M * block
    f = np.fft.rfftfreq(N, 1 / fs)[1:]
    lf, lv = np.log10(list(prof)), np.array(list(prof.values())) + extra_db
    L = np.interp(np.log10(f), lf, lv, left=lv[0], right=lv[-1])
    S = 2 * 10 ** (L / 10)
    df = fs / N
    amp = np.sqrt(S * df) / 2      # x = sum_k 2 Re(Z_k e^{...}), E|Z_k|^2 = 2 amp^2 -> var per bin = 4 amp^2 = S df
    Z = amp[None, :] * (rng.standard_normal((n_beacon, f.size)) + 1j * rng.standard_normal((n_beacon, f.size)))
    spec = np.concatenate([np.zeros((n_beacon, 1)), Z], axis=1)
    x = np.fft.irfft(spec, N, axis=1) * N
    i0 = (N - M) // 2
    return x[:, i0:i0 + M]


def iq_imbalance(y, image_dbc=-40.0, phase=0.0):
    nu = 10 ** (image_dbc / 20) * np.exp(1j * phase)
    return y + nu * np.conj(y)


def adc(y, bits=12, agc_step_db=1.0, headroom_db=3.0):
    """Per-beacon AGC (gain quantised to agc_step_db, peak at -headroom_db full scale) and uniform quantisation of I and
    Q to `bits`; returns the de-scaled samples. y: [n_b, M]."""
    peak = np.maximum(np.max(np.abs(np.concatenate([y.real, y.imag], axis=1)), axis=1), 1e-30)
    g_db = -20 * np.log10(peak) - headroom_db
    g_db = np.floor(g_db / agc_step_db) * agc_step_db
    g = 10 ** (g_db / 20)[:, None]
    q = 2.0 ** (1 - bits)
    z = y * g
    zq = (np.clip(np.round(z.real / q), -2 ** (bits - 1), 2 ** (bits - 1) - 1) +
          1j * np.clip(np.round(z.imag / q), -2 ** (bits - 1), 2 ** (bits - 1) - 1)) * q
    return zq / g


def sway_doppler(t_b, u_x, amp_m=0.005, freq_hz=1.0, phase=0.0):
    """Frequency shift (Hz) at one head from pole sway along the road: head velocity v_h = 2 pi f A cos(2 pi f t + phi)
    in x; the vehicle-to-head path shortens at rate -u_x v_h, u_x = x-component of the unit vector vehicle -> head."""
    v_h = 2 * np.pi * freq_hz * amp_m * np.cos(2 * np.pi * freq_hz * t_b + phase)
    return -u_x * v_h / LAMBDA


def apply_impairments(rng, y, t_b, u_x, imp, M=128, packet_s=1e-3):
    """Apply the receiver impairments to y [2, n_b, M] (after channel, offsets and noise): pole sway (per head), LO phase
    noise (independent per head), IQ imbalance (fixed random image phase per head), ADC with AGC. imp: dict from
    the constants file 'impairments' (nominal) or a sensitivity variant; None -> no impairments."""
    if imp is None:
        return y
    tau = tau_grid(M, packet_s)
    out = np.empty_like(y)
    sw = imp.get("pole_sway", {})
    for h in range(2):
        z = y[h]
        f_sw = sway_doppler(t_b, u_x[h], sw.get("amplitude_m", 0.0), sw.get("freq_hz", 1.0), rng.uniform(0, 2 * np.pi))
        z = z * np.exp(2j * np.pi * f_sw[:, None] * tau[None, :])
        z = z * np.exp(1j * phase_noise(rng, z.shape[0], M, packet_s, extra_db=imp.get("pn_extra_db", 0.0)))
        z = iq_imbalance(z, imp.get("iq_image_dBc", -40.0), rng.uniform(0, 2 * np.pi))
        out[h] = adc(z, imp.get("adc_bits", 12), imp.get("agc_step_dB", 1.0))
    return out
