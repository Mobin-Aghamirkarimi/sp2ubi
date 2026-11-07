"""Scatterer-mobility Doppler of the TR 37.885 V2V/V2R fast-fading model (own code; QuaDRiGa v2.8.1 does not implement it).

TR 37.885 clause 6.2.3: for the delayed paths, an extra
Doppler term 2·α_{n,m}·D_{n,m}/λ0 with D_{n,m} ~ U(−v_scatt, v_scatt), v_scatt the maximum vehicle speed of the layout,
and α_{n,m} ~ U(0, 1) (TR 38.901 clause 7.6.10 uses a Bernoulli α instead). The index (n, m) is per ray; whether the term is
applied per ray (Doppler spread inside a cluster) or per cluster (coherent shift) is decided by the official clause.
Both modes are implemented. The term is drawn once per pass and held fixed within the pass; the LoS path gets none.

Per-ray mode needs ray-level generator output; with cluster-level output only "cluster" mode can be applied."""
import numpy as np


def draw_extra_doppler(rng, n_clusters, n_rays, v_scatt, lam, alpha="uniform", p_bernoulli=0.2, granularity="ray"):
    """Extra Doppler (Hz) for every delayed ray, shape (n_clusters, n_rays); cluster mode repeats one value per cluster."""
    if granularity not in ("ray", "cluster"):
        raise ValueError(granularity)
    shape = (n_clusters, n_rays) if granularity == "ray" else (n_clusters, 1)
    d = rng.uniform(-v_scatt, v_scatt, shape)
    if alpha == "uniform":
        a = rng.uniform(0.0, 1.0, shape)
    elif alpha == "bernoulli":
        a = (rng.uniform(0.0, 1.0, shape) < p_bernoulli).astype(float)
    else:
        raise ValueError(alpha)
    nu = 2.0 * a * d / lam
    return np.broadcast_to(nu, (n_clusters, n_rays)).copy()


def apply(coeff, t, nu, los_index=None):
    """Rotate path coefficients coeff[..., path, time] by exp(j 2 pi nu_path t); nu has one value per path (flattened
    rays or clusters, in the same order as the path axis). The LoS path (los_index) is left unchanged."""
    nu = np.asarray(nu, float).ravel().copy()
    if los_index is not None:
        nu = np.insert(nu, los_index, 0.0) if nu.size == coeff.shape[-2] - 1 else nu
        nu[los_index] = 0.0
    if nu.size != coeff.shape[-2]:
        raise ValueError("one Doppler value per path required")
    return coeff * np.exp(2j * np.pi * nu[:, None] * np.asarray(t, float)[None, :])
