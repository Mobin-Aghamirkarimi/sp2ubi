"""Passive multistatic FDOA speed sensing of a vehicle's own V2X beacons.

Modules:
    geometry   -- site layout, vehicle trajectories, per-head Doppler
    channel    -- per-beacon, per-head narrowband baseband after data-aided waveform removal
    estimator  -- per-packet frequency-of-arrival, FDOA, joint pass fit, chi-square gate, Fisher bound
"""
