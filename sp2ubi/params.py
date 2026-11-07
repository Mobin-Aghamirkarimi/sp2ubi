"""Protocol parameters, kept in one place so the code and the paper's parameter table cannot drift apart."""
from dataclasses import dataclass


def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 3.3e24."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for q in small:
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def mac_prime(bits: int = 41, two_adicity: int = 16) -> int:
    """Smallest prime p > 2^(bits-1) with p = 1 (mod 2^two_adicity).

    p is both the HomMAC field and the BFV plaintext modulus t; the congruence makes p usable for SIMD batching
    for every ring dimension N <= 2^(two_adicity-1).
    """
    step = 1 << two_adicity
    k = ((1 << (bits - 1)) // step) + 1
    while not is_prime(k * step + 1):
        k += 1
    return k * step + 1


@dataclass(frozen=True)
class Params:
    """Risk path: one encrypted record per policy and aggregation interval of `interval_days` days (a contractual,
    policy-uniform parameter; default 1 = daily records; 31 = one record per monthly epoch). Record x = (d, off-peak,
    rush, weekend miles in 0.1-mile units, hard brakes, hard accelerations), d = insured days covered by the record
    (1..interval_days). The OBU enforces daily caps before summing. Each record is uploaded after an independent random
    delay uniform in [0, upload_window_s) after its interval ends (0 = immediate), over the vehicle's cellular (Uu) link
    directly to the TSP, a fresh connection per record, and re-sent until the TSP's receipt or the settlement deadline
    (RSUs do not relay risk records)."""
    n_features: int = 5            # risk-model features; coefficient position 0 is x_0 = D (insured days of the epoch)
    x_bits: int = 19               # epoch features < 2^x_bits (31 x 16383 = 507,873 < 2^19)
    w_bits: int = 22               # integer model weights, 0 <= w_i < 2^w_bits (largest relative rounding error 1.3e-4)
    p: int = mac_prime()           # HomMAC field = BFV plaintext modulus (41-bit prime, p ~ 2^40)
    epoch_days: int = 31           # longest calendar epoch (one calendar month for both paths)
    interval_days: int = 1         # aggregation interval a: 1 = daily records (default), 31 = monthly
    upload_window_s: int = 7 * 86400   # W: independent random upload delay per record
    settlement_grace_s: int = 7 * 86400  # contract parameter: settlement deadline = epoch end + W + grace (default 7 d);
    # the grace only extends re-sends after the drawn upload times, so the upload delays (W) are unchanged

    def settlement_deadline_s(self, epoch_len_days: int) -> int:
        """Seconds from the epoch start after which the TSP closes, shuffles and publishes the epoch's records."""
        return epoch_len_days * 86400 + self.upload_window_s + self.settlement_grace_s
    max_epoch_days: int = 31
    daily_mile_cap: int = (1 << 14) - 1   # per mileage feature and day, 0.1-mile units (1,638.3 miles)
    daily_event_cap: int = 4095           # hard brakes / hard accelerations per day
    nonce_bits: int = 40           # u: drawn uniformly in [1, p) per record and sealed; redrawn if already sealed in the epoch
    retry_window_s: int = 3600     # Uu re-send: next attempt uniform in [now, now + retry_window_s)

    @property
    def m(self) -> int:
        """Number of coefficient positions (features + x_0)."""
        return self.n_features + 1

    def records_per_epoch(self, D: int) -> int:
        """Number of records of a policy-epoch with D insured days."""
        return -(-D // self.interval_days)

    def record_bounds(self, w):
        """[S_min, S_max] of one record (1..interval_days insured days) with the actual weights."""
        return w[0], self.score_bounds(w, self.interval_days)[1]

    def epoch_caps(self, D: int):
        """Per-feature upper bounds of a record covering D insured days."""
        return [D] + [D * self.daily_mile_cap] * 3 + [D * self.daily_event_cap] * 2

    def score_bounds(self, w, D: int):
        """[S_min, S_max] of the epoch score S = w . x for D insured days with the actual weights (non-negative), so
        a decrypted S is decoded in [0, p) and range-checked against these bounds; S_max(31) must stay below p/2."""
        if any(wi < 0 for wi in w):
            raise ValueError("the decoding rule assumes non-negative weights")
        return w[0] * D, sum(wi * ci for wi, ci in zip(w, self.epoch_caps(D)))

    def check_bounds(self, w) -> None:
        """Re-run at every model update: the largest epoch score must stay below p/2."""
        if self.score_bounds(w, self.max_epoch_days)[1] >= self.p // 2:
            raise ValueError("epoch score bound >= p/2; lower the caps or enlarge p")
        if (1 << self.x_bits) <= max(self.epoch_caps(self.max_epoch_days)):
            raise ValueError("x_bits too small for the epoch caps")


DEFAULT = Params()
