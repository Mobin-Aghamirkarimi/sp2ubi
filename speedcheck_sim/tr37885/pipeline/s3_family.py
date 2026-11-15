"""Study 3 (urban estimator on newly simulated urban LOS sites): the site family U3 and its seeds. The family is
registered in pipeline/stage1_channels.py at run time (apply); no other module is changed.

U3: TR 37.885 urban LOS as the study-2 family U2 (one junction offset per site), family index 31. Per speed (10 and
15 m/s) every site has three roles: "ref", 30 survey crossings in traffic; "reff", 30 free-road survey crossings;
"test", 60 traffic crossings, which both survey designs share. Sites 30-39; sites 40-44 serve the snapshot-placement
check (pipeline/s3_snapshot_placement.py).

Seeds: digest = SHA-256(SEED_LABEL + DESIGN_COMMIT), where DESIGN_COMMIT identifies the commit that holds the urban
estimator and its acceptance criteria, made before any U3 site was drawn. The five seed bases (site, pass, SUMO,
impairments, bootstrap) are the first five big-endian 4-byte words of the digest. The study-2 constants are used
(SPEEDCHECK_CONSTANTS = config/study2_constants.json), as for U2."""
import hashlib

SEED_LABEL = "w-urban-fresh"
DESIGN_COMMIT = "4801d913f6e2035dad934cae19fd902fe75fc2ab"
TAG = "dev_s3"                      # stage-1 and stage-2 tag (provenance.check_tag admits dev_ tags outside study 1)
PLACE_TAG = "dev_s3_place"
FAMILY, IDX = "U3", 31
SITES, PLACE_SITES, PLACE_PASSES = (30, 40), (40, 45), 20
ROLES = (("ref", 30, True), ("reff", 30, False), ("test", 60, True))
SURVEY_ROLE = {"traffic": "ref", "free": "reff"}


def seeds():
    d = hashlib.sha256((SEED_LABEL + DESIGN_COMMIT).encode()).digest()
    w = [int.from_bytes(d[4 * i:4 * i + 4], "big") for i in range(5)]
    return dict(G1_site_base=w[0], G1_pass_base=w[1], sumo=w[2], impairments=w[3], bootstrap=w[4])


def apply():
    """Register U3 and its seeds in pipeline.stage1_channels (already on sys.path) and return that module."""
    from pipeline import stage1_channels as G
    from quadriga.driver import VEH_Z
    if FAMILY in G.FAMILIES:
        return G
    U2 = G.FAMILIES["U2"]
    G.FAMILIES[FAMILY] = dict(idx=IDX, cfg=U2["cfg"], scenario="urban", v_scatt=U2["v_scatt"], seedset=G.S2_SEEDSET,
                              tracks=[("P1", 10.0, ROLES, VEH_Z), ("P1", 15.0, ROLES, VEH_Z)])
    base = dict(G.C["seeds"], **seeds())
    other = G.seed_base

    def seed_base(fam):
        return base if fam == FAMILY else other(fam)

    G.seed_base = seed_base
    return G
