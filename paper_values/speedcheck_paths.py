"""Locations of the speed-check report generator and of its stored stage-3 inputs inside speedcheck_sim/."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "speedcheck_sim")
STORED = {
    "criteria_primary.json": ("results", "criteria_primary.json"),
    "inhouse_k4_sens_inhouse.json": ("results", "inhouse_k4_sens_inhouse.json"),
    "numbers_primary.json": ("results", "reports", "primary", "numbers_primary.json"),
    "make_reports.py": ("tr37885", "reports", "make_reports.py"),
}


def find(name):
    """Path of the stored file `name` of study 1 under speedcheck_sim/."""
    path = os.path.join(SIM, *STORED[name])
    if not os.path.isfile(path):
        sys.exit(f"missing {os.path.relpath(path, ROOT)}")
    return path


def import_make_reports():
    """The study-1 report generator (make_reports.py) with its package root on sys.path."""
    path = find("make_reports.py")
    sys.path.insert(0, os.path.dirname(path))
    sys.path.insert(0, os.path.dirname(os.path.dirname(path)))
    import make_reports
    return make_reports
