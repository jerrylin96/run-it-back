# SPDX-License-Identifier: Apache-2.0
#
# tests for analysis/ensemble_horizon.py, the shared core behind the
# 1000-member case-study instability/horizon diagnostics.
#
# What is worth locking here is not the arithmetic -- it is the handful of conventions
# that are silently wrong rather than loudly wrong, and each of them would corrupt a
# published figure without raising:
#
#   * the SPECTRUM/FIELD LEAD OFF-BY-ONE. The spectrum dump has R rows starting at the
#     first rolled step; the field dump has R+1 frames starting at the initial condition.
#     Off by one, every censoring decision lands on the wrong lead -- and the error is a
#     single 6 h step, exactly the size that looks like a plausible result.
#   * LONGITUDE WRAPPING. The comparison paper's box is written -126..-110; our grid runs
#     0..360. A box mask that silently matched nothing would return a NaN mean, but one
#     that matched the WRONG cells returns a perfectly plausible number for the wrong
#     part of the planet.
#   * MONOTONE CENSORING. A diverging trajectory can dip back under a ceiling for one
#     lead. Treating that as a recovery makes the survival curve non-monotone, which is
#     not a survival curve.
#   * cos(lat) AREA WEIGHTING. The comparison paper reduces on equal-area HEALPix pixels
#     with uniform weights; on our equiangular 0.25 deg grid the analog is cos(lat), and
#     an unweighted mean over a 38-55 N box is biased by several percent.
#   * POOL PROVENANCE. 250 dumps are only 1000 independent members if their noise seeds
#     differ. Seed 0 means "do not reseed", so a seed-0 task duplicates another's members.
#
# The fixture builders are module-level and importable, so the end-to-end CLI smoke uses
# the SAME synthetic dumps these tests assert on. torch-free (numpy + h5py), matching the
# module under test.

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "analysis"))

import ensemble_horizon as eh  # noqa: E402
from backward_fields_io import FieldDumpWriter  # noqa: E402

# a coarse global grid with the same ORIENTATION as ours: lat descending 90..-90, lon 0..360
H, W = 91, 180
LAT = np.linspace(90.0, -90.0, H)
LON = np.linspace(0.0, 360.0, W, endpoint=False)
CHANNELS = ["t2m", "z500"]
PNW_BOX = (-126.0, -110.0, 38.0, 55.0)


# ---------------------------------------------------------------------------
# fixture builders (shared with the end-to-end smoke)
# ---------------------------------------------------------------------------
def synthetic_truth(n_leads):
    """(L, H, W) plausible t2m-like field: a lat gradient plus a slowly drifting wave."""
    la = np.deg2rad(LAT)[:, None]
    lo = np.deg2rad(LON)[None, :]
    out = np.empty((n_leads, H, W), dtype=np.float32)
    for k in range(n_leads):
        out[k] = 288.0 - 45.0 * np.sin(la) ** 2 + 6.0 * np.cos(lo * 3 - 0.12 * k) * np.cos(la)
    return out


def write_case(dirpath, tag, seeds, n_members=4, n_leads=13, blowup=None, lam=1.35,
               channels=CHANNELS, step_hours=6, lmax=16, spec_decay=None):
    """Write a synthetic case: one field dump + one spectrum dump per seed.

    ``blowup``: {(seed, member): lead} -- from that field lead the member's field is
    multiplied away to a non-physical magnitude AND its spectral tail ratio is driven far
    past any ceiling, so both censoring detectors agree on the same member. Everything
    else grows like ``lam ** lead``, standing in for the |lambda| ~ 1.947 per step the real
    backward rollout shows.

    ``tag`` selects the filename scheme: "" gives the deprecated pre-tag
    ``backward_fields_seed%03d.h5``, a case tag gives ``backward_fields_<tag>_seed%03d.h5``.

    ``spec_decay``: when set, the member spectra lose power as exp(-rate * lead * (l/lmax)^2)
    -- a SCALE-SELECTIVE deficit, so small scales stop being recoverable before large ones.
    That is the shape diagnostic 4 exists to measure; the default (None) instead gives a
    uniform high-l EXCESS, which is the anti-diffusive shape the censoring criterion fires
    on. Both are real behaviours of this model at different leads, and the two fixtures
    keep the two detectors independently exercised.
    """
    import h5py

    os.makedirs(dirpath, exist_ok=True)
    truth = synthetic_truth(n_leads)
    blowup = blowup or {}
    if any(v < 1 for v in blowup.values()):
        raise ValueError("blow-up leads must be >= 1: lead 0 is the initial condition (pred == truth)")
    R = n_leads - 1
    stem = f"_{tag}" if tag else ""
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        fpath = os.path.join(dirpath, f"backward_fields{stem}_seed{seed:03d}.h5")
        wr = FieldDumpWriter(
            fpath, channels, 1, n_leads, LAT, LON, step_hours=step_hours,
            attrs={"config": "synthetic", "run_num": "test", "checkpoint_path": "none",
                   "ensemble_size": str(n_members), "n_members_dumped": str(n_members),
                   "noise_seed": str(seed), "input_noise_sigma": "1.0",
                   "pred_source": "all_members"},
            n_members=n_members)
        wr.set_anchor(0, 714)
        # member perturbation fields, fixed in space, amplified with lead
        pert = rng.standard_normal((n_members, len(channels), H, W)).astype(np.float32) * 0.35
        for lead in range(n_leads):
            pred = np.empty((n_members, len(channels), H, W), dtype=np.float32)
            for m in range(n_members):
                for c in range(len(channels)):
                    base = truth[lead] if channels[c] == "t2m" else truth[lead] * 190.0
                    # lead 0 is the INITIAL CONDITION: the rollout starts FROM truth, so the
                    # schema guarantees pred[:, m, 0] == truth[:, 0] for every member
                    # (backward_fields_io's file-layout note). A fixture that perturbed it
                    # would hide any consumer that wrongly assumes a non-zero spread there.
                    pred[m, c] = base if lead == 0 else base + pert[m, c] * (lam ** lead)
                bl = blowup.get((seed, m))
                if bl is not None and lead >= bl:
                    # a real backward blow-up runs to 1e34 and then NaN; float32 overflow to
                    # inf is exactly that, so the warning is expected, not a fixture defect
                    with np.errstate(over="ignore"):
                        pred[m] *= 10.0 ** (3 * (lead - bl + 1))
            tr = np.stack([truth[lead] if c == "t2m" else truth[lead] * 190.0 for c in channels])
            wr.write_frame(0, lead, pred, tr, timestamp=f"2021-06-{28 - lead // 4:02d}T12:00")
        wr.close()

        # spectrum dump: (R, C, cols, lmax); col 0 truth, col 1 ensemble mean, 2.. members.
        # Row k corresponds to FIELD LEAD k+1 -- the off-by-one under test.
        mean = np.zeros((R, len(channels), n_members + 2, lmax))
        ell = np.arange(lmax) + 1.0
        base_spec = ell ** -3.0
        for k in range(R):
            lead = k + 1
            for c in range(len(channels)):
                mean[k, c, 0] = base_spec
                for m in range(n_members):
                    if spec_decay is None:
                        ratio = np.ones(lmax)
                        ratio[lmax // 2:] = 1.15 ** lead      # gentle anti-diffusive excess
                    else:
                        # scale-selective blur: the deficit deepens with lead and with l
                        ratio = np.exp(-float(spec_decay) * lead * (ell / lmax) ** 2)
                    bl = blowup.get((seed, m))
                    if bl is not None and lead >= bl:
                        ratio[lmax // 2:] = 1e4 * (lead - bl + 1)
                    mean[k, c, 2 + m] = base_spec * ratio
                mean[k, c, 1] = mean[k, c, 2:].mean(axis=0)
        spath = os.path.join(dirpath, f"backward_spectrum{stem}_seed{seed:03d}.h5")
        with h5py.File(spath, "w") as f:
            f.create_dataset("mean", data=mean)
            f.create_dataset("std", data=np.zeros_like(mean))
            f.create_dataset("lead_time", data=np.arange(1, R + 1, dtype=np.float64))
            f.create_dataset("channel", data=np.array([c.encode() for c in channels], dtype="S16"))
            f.create_dataset("lat", data=LAT)
            f.create_dataset("lon", data=LON)
    return dirpath


def write_land_mask(path):
    """A land-sea-mask invariant in makani's contract (key LSM at [0,:,:]). Land is a
    single lon band covering the western half of the PNW box, so the mask provably
    CHANGES which cells the box selects rather than being a no-op."""
    import h5py

    lsm = np.zeros((H, W), dtype=np.float32)
    lon360 = np.mod(LON, 360.0)
    lsm[:, (lon360 >= 234.0) & (lon360 <= 242.0)] = 1.0
    with h5py.File(path, "w") as f:
        f.create_dataset("LSM", data=lsm[None, :, :])
    return path


# ---------------------------------------------------------------------------
# discovery / provenance
# ---------------------------------------------------------------------------
def test_parse_seed_handles_both_filename_schemes():
    assert eh.parse_seed("backward_fields_seed007.h5") == 7                      # deprecated, pre-tag
    assert eh.parse_seed("backward_fields_20210628T1200_seed250.h5") == 250      # case-tagged
    assert eh.parse_seed("/a/seed999/backward_fields_20220930T1800_seed042.h5") == 42  # dir decoy loses
    with pytest.raises(ValueError):
        eh.parse_seed("backward_fields.h5")


def test_discover_orders_by_seed_and_rejects_empty(tmp_path):
    d = write_case(str(tmp_path / "c"), "20210628T1200", seeds=[3, 1, 2])
    got = eh.discover(os.path.join(d, "backward_fields_*_seed*.h5"))
    assert [s for s, _ in got] == [1, 2, 3]
    assert eh.parse_seed(got[0][1]) == 1
    with pytest.raises(SystemExit):
        eh.discover(os.path.join(d, "no_such_*.h5"))
    with pytest.raises(SystemExit):
        eh.discover(os.path.join(d, "backward_fields_*_seed*.h5"), expect=99)


def test_provenance_rejects_seed_zero_and_duplicate_draws():
    ok = {1: {"noise_seed": "1", "pred_source": "all_members", "ensemble_size": "4", "n_members_dumped": "4"},
          2: {"noise_seed": "2", "pred_source": "all_members", "ensemble_size": "4", "n_members_dumped": "4"}}
    rep = eh.check_pool_provenance(ok)
    assert rep["n_distinct_noise_seeds"] == 2 and not rep["problems"]

    # seed 0 = "do not reseed": these members duplicate another task's, inflating the pool
    with pytest.raises(SystemExit):
        eh.check_pool_provenance({0: dict(ok[1], noise_seed="0")})
    # a dump that kept only the ensemble mean is not a member pool at all
    with pytest.raises(SystemExit):
        eh.check_pool_provenance({1: dict(ok[1], pred_source="ensemble_mean")})
    # filename says one seed, the file says another -> the pool's independence is unproven
    with pytest.raises(SystemExit):
        eh.check_pool_provenance({1: dict(ok[1], noise_seed="7")})
    # non-strict downgrades to a report
    rep = eh.check_pool_provenance({1: dict(ok[1], noise_seed="7")}, strict=False)
    assert rep["problems"]


# ---------------------------------------------------------------------------
# domain reduction
# ---------------------------------------------------------------------------
def test_box_mask_wraps_longitude_both_conventions():
    neg = eh.box_mask(LAT, LON, PNW_BOX)                       # -126..-110 against a 0..360 grid
    pos = eh.box_mask(LAT, LON, (234.0, 250.0, 38.0, 55.0))    # the same box, already wrapped
    assert neg.any()
    np.testing.assert_array_equal(neg, pos)
    # exactly the cells inside the box, and nothing else
    lo, la = np.meshgrid(np.mod(LON, 360.0), LAT)
    np.testing.assert_array_equal(neg, (lo >= 234.0) & (lo <= 250.0) & (la >= 38.0) & (la <= 55.0))


def test_box_mask_crossing_the_prime_meridian_is_a_union():
    m = eh.box_mask(LAT, LON, (-10.0, 10.0, -5.0, 5.0))
    lo = np.mod(LON, 360.0)
    assert m[:, lo <= 10.0].any() and m[:, lo >= 350.0].any()
    assert not m[:, (lo > 10.0) & (lo < 350.0)].any()


def test_domain_weights_are_cos_lat_and_respect_the_land_mask(tmp_path):
    lsm = eh.load_land_fraction(write_land_mask(str(tmp_path / "lsm.h5")))
    w_all = eh.domain_weights(LAT, LON, PNW_BOX)
    w_land = eh.domain_weights(LAT, LON, PNW_BOX, lsm, 0.5, "land")
    w_sea = eh.domain_weights(LAT, LON, PNW_BOX, lsm, 0.5, "sea")
    assert (w_land > 0).sum() < (w_all > 0).sum()               # the mask really removes cells
    np.testing.assert_allclose(w_land + w_sea, w_all)           # land and sea partition the box
    # weights are cos(lat) where selected -- an unweighted mean over 38-55 N is a different number
    rows = np.where((w_all > 0).any(axis=1))[0]
    for r in rows:
        np.testing.assert_allclose(w_all[r, w_all[r] > 0], np.cos(np.deg2rad(LAT[r])), rtol=1e-12)
    with pytest.raises(SystemExit):                             # empty domain fails loud, not NaN
        eh.domain_weights(LAT, LON, (0.0, 1.0, 88.0, 89.0), lsm, 0.5, "land")


def test_weighted_mean_matches_an_explicit_cos_lat_average():
    w = eh.domain_weights(LAT, LON, PNW_BOX)
    f = synthetic_truth(3)
    got = eh.weighted_mean(f, w)
    assert got.shape == (3,)
    for k in range(3):
        np.testing.assert_allclose(got[k], (f[k] * w).sum() / w.sum(), rtol=1e-12)
    # leading axes are preserved, so members and truth reduce through ONE weight array
    assert eh.weighted_mean(np.broadcast_to(f, (5, 3, H, W)), w).shape == (5, 3)


# ---------------------------------------------------------------------------
# censoring / survivors
# ---------------------------------------------------------------------------
def test_censoring_is_monotone_through_a_transient_dip():
    # a diverging member that dips back under the ceiling at index 4 has not recovered
    series = [1.0, 2.0, 5.0, 12.0, 3.0, 40.0]
    k = eh.first_failure(series, 10.0)
    assert k == 3
    alive = eh.make_monotone_alive(k, len(series))
    np.testing.assert_array_equal(alive, [True, True, True, False, False, False])
    assert np.all(np.diff(alive.astype(int)) <= 0)
    assert eh.first_failure([1.0, 2.0], 10.0) is None
    assert eh.first_failure([1.0, float("nan"), 2.0], 1e9) == 1       # non-finite is failure
    np.testing.assert_array_equal(eh.make_monotone_alive(None, 3), [True, True, True])


def test_combine_failures_takes_the_earliest_detector():
    assert eh.combine_failures(5, 2) == 2
    assert eh.combine_failures(None, 7) == 7
    assert eh.combine_failures(None, None) is None
    assert eh.first_true([False, False, True, False]) == 2
    assert eh.first_true([False, False]) is None


def test_survivor_stats_ignores_dead_members_and_survives_an_empty_lead():
    values = np.array([[1.0, 2.0, 1e9], [1.0, 4.0, 1e9], [1.0, 6.0, 1e9], [1.0, 8.0, 1e9]])
    alive = np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=bool)
    st = eh.survivor_stats(values, alive)
    np.testing.assert_allclose(st["survival"], [1.0, 0.5, 0.0])
    np.testing.assert_array_equal(st["n_alive"], [4, 2, 0])
    np.testing.assert_allclose(st["q"][1, :2], [1.0, 3.0])       # median over survivors only
    assert np.isnan(st["q"][:, 2]).all()                          # fully censored lead -> NaN, no crash
    with pytest.raises(ValueError):
        eh.survivor_stats(values, alive[:2])


def test_horizon_at_interpolates_the_crossing():
    x = np.array([0.0, 6.0, 12.0, 18.0])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    assert eh.horizon_at(x, y, 0.5) == pytest.approx(9.0)          # halfway between 6 and 12
    assert eh.horizon_at(x, np.ones(4), 0.5) is None               # never crosses
    # rising series crosses 0.5 between x=6 (y=0.2) and x=12 (y=0.8): 6 + 0.3*6/0.6 = 9
    assert eh.horizon_at(x, np.array([0.0, 0.2, 0.8, 1.0]), 0.5, decreasing=False) == pytest.approx(9.0)


def test_exponential_fit_recovers_a_known_growth_rate():
    hours = np.arange(0, 61, 6.0)
    rate = np.log(1.947) / 6.0                                     # the recorded |lambda| per 6 h step
    slope, icept, n = eh.exponential_fit(hours, 0.4 * np.exp(rate * hours))
    assert slope == pytest.approx(rate, rel=1e-9)
    assert np.exp(slope * 6.0) == pytest.approx(1.947, rel=1e-9)
    assert np.log(2.0) / slope == pytest.approx(6.23, abs=0.02)    # the abstract's ~6.2 h doubling
    assert n == len(hours)
    # censored (NaN) and non-positive samples are dropped rather than poisoning the fit
    y = 0.4 * np.exp(rate * hours)
    y[-3:] = np.nan
    assert eh.exponential_fit(hours, y)[0] == pytest.approx(rate, rel=1e-9)
    assert eh.exponential_fit(hours, np.zeros_like(hours))[2] == 0


# ---------------------------------------------------------------------------
# spectra -- the lead off-by-one and the member-column layout
# ---------------------------------------------------------------------------
def test_spectrum_rows_map_to_field_lead_k_plus_one(tmp_path):
    d = write_case(str(tmp_path / "c"), "20210628T1200", seeds=[1], n_leads=13,
                   blowup={(1, 2): 5})
    spec = eh.load_member_spectra(os.path.join(d, "backward_spectrum_20210628T1200_seed001.h5"))
    assert spec["n_members"] == 4
    assert spec["mean"].shape[0] == 12                      # R rows for R+1 field frames
    np.testing.assert_array_equal(spec["lead_index"], np.arange(1, 13))

    tail = eh.tail_ratio_members(spec, "t2m")               # (M, R)
    assert tail.shape == (4, 12)
    # member 2 was blown up from FIELD lead 5; its spectrum row is therefore index 4
    k = eh.first_failure(tail[2], eh.TAIL_CEIL_DEFAULT)
    assert k == 4
    assert k + 1 == 5                                        # ... which maps back to field lead 5
    for m in (0, 1, 3):
        assert eh.first_failure(tail[m], eh.TAIL_CEIL_DEFAULT) is None


def test_member_columns_are_offset_by_two_and_ordered(tmp_path):
    d = write_case(str(tmp_path / "c"), "", seeds=[4], n_members=4,
                   blowup={(4, 0): 3, (4, 3): 9})
    spec = eh.load_member_spectra(os.path.join(d, "backward_spectrum_seed004.h5"))
    tail = eh.tail_ratio_members(spec, "t2m")
    # distinct blow-up leads per member prove the columns are not transposed or shifted
    assert eh.first_failure(tail[0], eh.TAIL_CEIL_DEFAULT) == 2
    assert eh.first_failure(tail[3], eh.TAIL_CEIL_DEFAULT) == 8
    assert eh.first_failure(tail[1], eh.TAIL_CEIL_DEFAULT) is None
    with pytest.raises(SystemExit):
        eh.tail_ratio_members(spec, "not_a_channel")


def test_banded_ratio_splits_the_spectrum_without_double_counting(tmp_path):
    d = write_case(str(tmp_path / "c"), "", seeds=[1])
    spec = eh.load_member_spectra(os.path.join(d, "backward_spectrum_seed001.h5"))
    edges = eh.band_edges(spec["lmax"], 3)
    assert edges[0] >= 1 and edges[-1] == spec["lmax"]        # l=0 excluded, lmax covered
    assert np.all(np.diff(edges) > 0)                          # half-open, strictly increasing
    banded = eh.banded_ratio_members(spec, "t2m", edges)
    assert banded.shape == (spec["n_members"], spec["mean"].shape[0], len(edges) - 1)
    # the fixture's excess lives only above lmax//2, so the largest-scale band stays at 1
    np.testing.assert_allclose(banded[:, -1, 0], 1.0, rtol=1e-9)
    assert np.all(banded[:, -1, -1] > 1.0)


def test_deterministic_spectrum_is_rejected_rather_than_read_as_one_member(tmp_path):
    import h5py

    p = str(tmp_path / "det.h5")
    with h5py.File(p, "w") as f:
        f.create_dataset("mean", data=np.ones((4, 2, 2, 16)))   # only [truth, pred]
        f.create_dataset("channel", data=np.array([b"t2m", b"z500"], dtype="S16"))
    with pytest.raises(SystemExit):
        eh.load_member_spectra(p)


# ---------------------------------------------------------------------------
# CLI-level integration -- the pooled path end to end
#
# The unit tests above pin each convention in isolation. These pin the COMPOSITION:
# a fixture whose divergence leads are known exactly must come back out of the full
# pooling + censoring pipeline as exactly that survival curve. If the spectrum/field
# off-by-one ever slips, or the pool concatenates on the wrong axis, or a censored
# member leaks into a survivor statistic, this is what notices -- none of which shows up
# as an exception, only as a plausible-looking wrong figure.
# ---------------------------------------------------------------------------
import argparse  # noqa: E402

sys.path.insert(0, os.path.join(_ROOT, "analysis"))
import plot_ensemble_horizon as peh  # noqa: E402
import plot_spectral_horizon as psh  # noqa: E402


def _fields_args(**over):
    base = dict(channel="t2m", anchor=0, box=PNW_BOX, land_mask=None, land_threshold=0.5,
                mask_want="land", criterion="auto", tail_ceil=eh.TAIL_CEIL_DEFAULT, tail_l0=None,
                range_lo=150.0, range_hi=350.0, spread_stride=1, no_spread=False, max_files=None,
                progress=0, allow_bad_provenance=False, sensitivity_ceilings=[3.0, 10.0],
                raw_geopotential=False)
    base.update(over)
    return argparse.Namespace(**base)


def _expected_survival(blowup, seeds, n_members, n_leads):
    """Survival implied by the fixture's blow-up table, computed independently of the code
    under test: a member is alive at lead k iff it has no blow-up lead <= k."""
    fails = [blowup.get((s, m)) for s in seeds for m in range(n_members)]
    tot = float(len(fails))
    return np.array([sum(1 for f in fails if f is None or f > k) / tot for k in range(n_leads)])


def test_pooled_survival_matches_the_fixtures_known_divergence_leads(tmp_path):
    seeds, E, L = [1, 2, 3, 4], 4, 13
    blow = {(1, 0): 3, (1, 2): 7, (2, 1): 5, (3, 3): 11, (4, 0): 9, (4, 1): 9}
    d = write_case(str(tmp_path / "c"), "20210628T1200", seeds=seeds, n_members=E,
                   n_leads=L, blowup=blow)
    got = peh._load_case("case", os.path.join(d, "backward_fields_*_seed*.h5"),
                         os.path.join(d, "backward_spectrum_*_seed*.h5"),
                         _fields_args(), lambda m: None)
    assert got["criterion"] == "tail"                       # auto-resolves to tail when spectra exist
    assert got["scalar"].shape == (len(seeds) * E, L)       # pooled on the member axis, not stacked wrong
    np.testing.assert_array_equal(got["seeds"], np.repeat(seeds, E))
    np.testing.assert_allclose(got["stats"]["survival"], _expected_survival(blow, seeds, E, L))


def test_range_and_tail_criteria_agree_on_the_same_fixture(tmp_path):
    seeds, E, L = [1, 2], 4, 13
    blow = {(1, 0): 4, (2, 3): 8}
    d = write_case(str(tmp_path / "c"), "", seeds=seeds, n_members=E, n_leads=L, blowup=blow)
    fg = os.path.join(d, "backward_fields_seed*.h5")
    sg = os.path.join(d, "backward_spectrum_seed*.h5")
    by_tail = peh._load_case("t", fg, sg, _fields_args(criterion="tail"), lambda m: None)
    by_range = peh._load_case("r", fg, None, _fields_args(criterion="range"), lambda m: None)
    by_either = peh._load_case("e", fg, sg, _fields_args(criterion="either"), lambda m: None)
    exp = _expected_survival(blow, seeds, E, L)
    # the field detector and the spectral detector are wired to DIFFERENT files with
    # different lead conventions; agreeing on the same censoring is what proves the
    # spectrum-row -> field-lead mapping is right rather than merely self-consistent
    for got in (by_tail, by_range, by_either):
        np.testing.assert_allclose(got["stats"]["survival"], exp)
    np.testing.assert_array_equal(by_tail["alive"], by_range["alive"])


def test_pooled_domain_mean_matches_an_independent_cos_lat_masked_average(tmp_path):
    d = write_case(str(tmp_path / "c"), "", seeds=[1], n_members=2, n_leads=5)
    lsm_path = write_land_mask(str(tmp_path / "lsm.h5"))
    got = peh._load_case("c", os.path.join(d, "backward_fields_seed*.h5"), None,
                         _fields_args(criterion="range", land_mask=lsm_path), lambda m: None)
    # rebuild the expected ERA5 domain mean from the fixture's own truth, by hand
    w = eh.domain_weights(LAT, LON, PNW_BOX, eh.load_land_fraction(lsm_path), 0.5, "land")
    truth = synthetic_truth(5)
    expect = np.array([(truth[k] * w).sum() / w.sum() for k in range(5)])
    np.testing.assert_allclose(got["truth_scalar"], expect, rtol=1e-6)
    # the mask really bit: the land-only mean differs from the whole-box mean
    w_all = eh.domain_weights(LAT, LON, PNW_BOX)
    assert not np.allclose(expect, [(truth[k] * w_all).sum() / w_all.sum() for k in range(5)])
    # lead 0 is the initial condition: every member equals truth there, so spread starts at 0
    np.testing.assert_allclose(got["scalar"][:, 0], expect[0], rtol=1e-6)


def test_spread_is_the_area_weighted_ensemble_std_over_survivors(tmp_path):
    d = write_case(str(tmp_path / "c"), "", seeds=[1, 2], n_members=4, n_leads=6)
    args = _fields_args(criterion="range", box=None)
    got = peh._load_case("c", os.path.join(d, "backward_fields_seed*.h5"), None, args, lambda m: None)
    # recompute independently from the raw dumps
    from backward_fields_io import load_channel_members
    preds = [load_channel_members(os.path.join(d, f"backward_fields_seed{s:03d}.h5"), "t2m")["pred"]
             for s in (1, 2)]
    allp = np.concatenate(preds, axis=0).astype(np.float64)          # (M, L, H, W)
    w = eh.domain_weights(LAT, LON, None)
    var = allp.var(axis=0, ddof=1)                                    # (L, H, W)
    expect = np.sqrt((var * w).sum(axis=(-2, -1)) / w.sum())
    np.testing.assert_allclose(got["spread"], expect, rtol=1e-5)


def test_spread_stride_is_a_subsample_not_a_different_statistic(tmp_path):
    d = write_case(str(tmp_path / "c"), "", seeds=[1], n_members=4, n_leads=4)
    fg = os.path.join(d, "backward_fields_seed*.h5")
    full = peh._load_case("a", fg, None, _fields_args(criterion="range", box=None), lambda m: None)
    strided = peh._load_case("b", fg, None,
                             _fields_args(criterion="range", box=None, spread_stride=3), lambda m: None)
    # lead 0 is the IC (pred == truth for every member), so the spread starts at exactly 0
    # regardless of the stride -- and a coarser sample tracks the full grid thereafter
    assert full["spread"][0] == pytest.approx(0.0, abs=1e-9)
    assert strided["spread"][0] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(strided["spread"][1:], full["spread"][1:], rtol=0.25)


def test_a_seed_with_no_matching_spectrum_fails_loud(tmp_path):
    d = write_case(str(tmp_path / "c"), "", seeds=[1, 2], n_members=2, n_leads=5)
    os.remove(os.path.join(d, "backward_spectrum_seed002.h5"))
    with pytest.raises(SystemExit, match="no spectrum dump"):
        peh._load_case("c", os.path.join(d, "backward_fields_seed*.h5"),
                       os.path.join(d, "backward_spectrum_seed*.h5"),
                       _fields_args(criterion="tail"), lambda m: None)


def test_mismatched_dump_depth_in_one_pool_fails_loud(tmp_path):
    d = str(tmp_path / "c")
    write_case(d, "", seeds=[1], n_members=2, n_leads=5)
    write_case(d, "", seeds=[2], n_members=2, n_leads=7)      # same glob, different depth
    with pytest.raises(SystemExit, match="leads"):
        peh._load_case("c", os.path.join(d, "backward_fields_seed*.h5"), None,
                       _fields_args(criterion="range"), lambda m: None)


def _spec_args(**over):
    base = dict(max_files=None, n_bands=3, l_min=1, censor_channel=None,
                tail_ceil=eh.TAIL_CEIL_DEFAULT, tail_l0=None, step_hours=6.0,
                provenance=None, allow_bad_provenance=False)
    base.update(over)
    return argparse.Namespace(**base)


def test_spectral_case_survival_and_lead_hours(tmp_path):
    seeds, E, L = [1, 2, 3], 4, 13
    blow = {(1, 0): 3, (2, 2): 6, (3, 1): 10}
    d = write_case(str(tmp_path / "c"), "20220930T1800", seeds=seeds, n_members=E,
                   n_leads=L, blowup=blow)
    got = psh._load_case("ian", os.path.join(d, "backward_spectrum_*_seed*.h5"),
                         ["t2m"], _spec_args(), lambda m: None)
    assert got["n_members"] == len(seeds) * E
    # spectrum space has L-1 rows and its first row is field lead 1 -> -6 h, not -0 h
    np.testing.assert_allclose(got["leads"], np.arange(1, L) * 6.0)
    # survival in spectrum-row space: row k is field lead k+1
    exp_field = _expected_survival(blow, seeds, E, L)
    np.testing.assert_allclose(got["survival"], exp_field[1:])


def test_spectral_horizon_shrinks_with_wavenumber_on_a_blurring_fixture(tmp_path):
    d = write_case(str(tmp_path / "c"), "", seeds=[1, 2], n_members=4, n_leads=31,
                   lmax=64, spec_decay=0.3)
    got = psh._load_case("c", os.path.join(d, "backward_spectrum_seed*.h5"), ["z500"],
                         _spec_args(n_bands=4), lambda m: None)
    hs = []
    for b in range(len(got["edges"]) - 1):
        med = eh.survivor_stats(got["banded"]["z500"][:, :, b], got["alive"])["q"][1]
        hs.append(eh.horizon_with_status(got["leads"], med, 0.75))
    # Ordering is asserted over the BOUNDED bands only, and each one's status decides
    # which end it is bounded at. Filtering on `h is not None` -- as this test used to --
    # silently pooled "never crossed" (deepest) with "already past at the first sample"
    # (shallowest), i.e. it encoded the very inversion the plot was making, which is why
    # it passed over a live bug.
    ordered = [v for v, s in hs if s in (eh.CROSSED, eh.BEFORE_FIRST)]
    assert len(ordered) >= 2, f"fixture produced too few bounded bands to test the ordering: {hs}"
    # the physics claim: a smaller scale stops being recoverable EARLIER
    assert all(a >= b for a, b in zip(ordered, ordered[1:])), f"horizons not decreasing with l: {hs}"
    # ... and the largest-scale band should be the deepest: either it never crosses, or it
    # crosses no earlier than any bounded band. It must NOT be BEFORE_FIRST.
    v0, s0 = hs[0]
    assert s0 != eh.BEFORE_FIRST, f"largest-scale band read as already-blurred at the first lead: {hs}"
    assert s0 == eh.NEVER or v0 >= max(ordered)


def test_a_band_already_past_the_level_at_the_first_lead_is_not_read_as_surviving():
    """The regression for the spectral horizon INVERSION.

    Spectra carry no lead-0 row (read_spectrum: lead_index starts at 1), so a small scale
    the model has already blurred by -6 h is below the ratio at the FIRST sample. That used
    to return None, which the figure pinned to the rollout depth and the text printed as
    "never" -- ranking the fastest-dying scale as the longest-surviving, the exact reverse
    of diagnostic 4's claim.
    """
    leads = np.array([6.0, 12.0, 18.0, 24.0])
    dead = np.array([0.40, 0.30, 0.20, 0.10])        # already blurred at the first sample
    alive = np.array([0.99, 0.95, 0.90, 0.88])       # never reaches the level

    v, s = eh.horizon_with_status(leads, dead, 0.75)
    assert s == eh.BEFORE_FIRST
    assert v == pytest.approx(6.0), "BEFORE_FIRST reports the first sampled lead as a bound"

    v2, s2 = eh.horizon_with_status(leads, alive, 0.75)
    assert (v2, s2) == (None, eh.NEVER)

    # the two must not collapse to one answer -- that collapse WAS the bug
    assert s != s2
    # and the shallow one must order ahead of a real crossing
    v3, s3 = eh.horizon_with_status(leads, np.array([0.9, 0.8, 0.7, 0.6]), 0.75)
    assert s3 == eh.CROSSED and v3 > v


def test_horizon_at_gives_a_bound_not_the_opposite_answer_when_it_starts_past():
    """The single-number wrapper cannot express BEFORE_FIRST, but it must not report the
    opposite conclusion: it returns the first lead (a bound), never None (deeper than the
    rollout)."""
    leads = np.array([6.0, 12.0, 18.0])
    assert eh.horizon_at(leads, np.array([0.4, 0.3, 0.2]), 0.75) == pytest.approx(6.0)
    # survival series start at 1.0 by construction, so the wrapper is unaffected there
    assert eh.horizon_at(leads, np.array([1.0, 1.0, 1.0]), 0.5) is None


def test_full_zonal_box_selects_every_longitude():
    """`--box -180 180 30 60` and `0 360 30 60` wrap to west == east. The general
    comparison then selects the single meridian lon == west, which is not empty, so
    nothing downstream raises and the run silently reduces over a 1-D slice."""
    lat = np.linspace(90, -90, 7)
    lon = np.linspace(0, 360, 8, endpoint=False)
    for box in [(-180.0, 180.0, 30.0, 60.0), (0.0, 360.0, 30.0, 60.0)]:
        m = eh.box_mask(lat, lon, box)
        assert m.any(axis=0).sum() == len(lon), f"box {box} collapsed to {m.any(axis=0).sum()} lons"
        assert m.sum() == len(lon) * int(((lat >= 30.0) & (lat <= 60.0)).sum())
    # a genuine sub-span still excludes longitudes, and a wrapping box still unions
    assert eh.box_mask(lat, lon, (0.0, 90.0, -90.0, 90.0)).any(axis=0).sum() < len(lon)
    assert eh.box_mask(lat, lon, (-30.0, 30.0, -90.0, 90.0)).any(axis=0).sum() < len(lon)


def test_ordering_verdict_keeps_a_never_band_in_the_sequence():
    """A NEVER band between two measured crossings outlives its smaller-scale neighbour,
    which VIOLATES the predicted ordering. Dropping it -- as the first fix did -- left the
    remaining bands reading monotone and printed the prediction as confirmed."""
    vals = [21.0, 13.5, None, 7.0]
    stats = [eh.CROSSED, eh.CROSSED, eh.NEVER, eh.CROSSED]
    verdict, shown = psh.ordering_verdict(vals, stats, deepest=180.0)
    assert "NOT monotone" in verdict, verdict
    assert shown == ["21", "14", ">180", "7"]
    # the same bands WITHOUT the offending one are monotone -- which is exactly why
    # dropping it was silently wrong
    v2, _ = psh.ordering_verdict([21.0, 13.5, 7.0], [eh.CROSSED] * 3, deepest=180.0)
    assert "SHRINKS" in v2


def test_ordering_verdict_will_not_confirm_the_prediction_without_measurements():
    """Bands all pinned to the same bound are a flat sequence, trivially 'monotone'. That
    must not print as the falsifiable prediction confirmed."""
    verdict, shown = psh.ordering_verdict([6.0] * 5, [eh.BEFORE_FIRST] * 5, deepest=180.0)
    assert verdict.startswith("UNRESOLVED"), verdict
    assert shown == ["<6"] * 5
    # one measured crossing is still not two
    v1, _ = psh.ordering_verdict([6.0, 6.0, 40.0], [eh.BEFORE_FIRST, eh.BEFORE_FIRST, eh.CROSSED], 180.0)
    assert v1.startswith("UNRESOLVED"), v1
    # two measured crossings resolve it, and the bound still orders last
    v2, _ = psh.ordering_verdict([40.0, 20.0, 6.0], [eh.CROSSED, eh.CROSSED, eh.BEFORE_FIRST], 180.0)
    assert "SHRINKS" in v2, v2


def test_horizon_rendering_never_uses_a_signed_inequality():
    """'<= -6 h' on a signed axis admits -96 h -- deeper, the opposite of the intended
    bound, and the same inversion class this module was fixed for. Bounds are quoted as a
    depth in hours BACK, and each names the lead bounding it."""
    band = {"horizon_h": {"0.5": 6.0}, "horizon_status": {"0.5": eh.BEFORE_FIRST}}
    assert psh._fmt_horizon(band, "0.5", 180.0) == "< 6 h back"
    never = {"horizon_h": {"0.5": None}, "horizon_status": {"0.5": eh.NEVER}}
    assert psh._fmt_horizon(never, "0.5", 180.0) == "> 180 h back"
    crossed = {"horizon_h": {"0.5": 42.0}, "horizon_status": {"0.5": eh.CROSSED}}
    assert psh._fmt_horizon(crossed, "0.5", 180.0) == "-42 h"
    for txt in ("< 6 h back", "> 180 h back"):
        assert "-" not in txt, "a bound must not be quoted as a signed coordinate"


# ---------------------------------------------------------------------------
# Regressions from the Aug 2026 review of the diagnostics build. Each locks a path that
# returned a plausible-looking WRONG number rather than failing, which is the only kind of
# defect this analysis cannot survive: a figure that looks fine is never re-derived.
# ---------------------------------------------------------------------------
def test_spectral_provenance_is_unchecked_without_the_field_dumps(tmp_path):
    """Diagnostic 4 must SAY its pool is unverified rather than imply it was checked.

    The spectrum dumps carry no attrs at all (makani's SpectrumAverageBuffer writes only
    datasets), so the seed-0 / duplicate-draw check diagnostics 1-3 run cannot be run here.
    """
    d = write_case(str(tmp_path / "c"), "20220930T1800", seeds=[1, 2], n_members=3, n_leads=9)
    lines = []
    psh._load_case("ian", os.path.join(d, "backward_spectrum_*_seed*.h5"),
                   ["t2m"], _spec_args(), lines.append)
    assert any("provenance UNCHECKED" in m for m in lines), lines


def test_spectral_provenance_rejects_a_seed_zero_pool_via_the_field_dumps(tmp_path):
    """--provenance points at the sibling field dumps, whose attrs DO carry the seed.

    noise_seed 0 means "do not reseed", so such a task's members duplicate another's; the
    pool is then not the independent draws the horizon is quoted off.
    """
    d = str(tmp_path / "c")
    write_case(d, "20220930T1800", seeds=[1, 2], n_members=3, n_leads=9)
    # rewrite one dump's recorded seed to 0, leaving the filename alone: exactly the case
    # eh.discover cannot see, because it only checks filename-seed uniqueness.
    import h5py
    with h5py.File(os.path.join(d, "backward_fields_20220930T1800_seed002.h5"), "r+") as f:
        f.attrs["noise_seed"] = "0"
    args = _spec_args(provenance=[f"ian={os.path.join(d, 'backward_fields_*_seed*.h5')}"])
    with pytest.raises(SystemExit, match="noise_seed"):
        psh._load_case("ian", os.path.join(d, "backward_spectrum_*_seed*.h5"),
                       ["t2m"], args, lambda m: None)


def test_spectral_provenance_accepts_a_clean_pool(tmp_path):
    d = str(tmp_path / "c")
    write_case(d, "20220930T1800", seeds=[1, 2, 3], n_members=3, n_leads=9)
    lines = []
    args = _spec_args(provenance=[f"ian={os.path.join(d, 'backward_fields_*_seed*.h5')}"])
    psh._load_case("ian", os.path.join(d, "backward_spectrum_*_seed*.h5"),
                   ["t2m"], args, lines.append)
    assert any("provenance: 3 files, 3 distinct" in m for m in lines), lines
    assert not any("UNCHECKED" in m for m in lines), lines


def test_sensitivity_uses_the_cases_own_criterion_not_tail_alone(tmp_path):
    """Under 'either' the sensitivity curve at the shared ceiling must EQUAL the headline one.

    It used to be recomputed tail-only, so it sat above the curve it was drawn beside --
    including at the ceiling where the two are the same threshold -- and the 'near-parallel
    curves' argument was read off a different censoring rule from the number it defends.
    """
    import h5py
    d = str(tmp_path / "c")
    write_case(d, "20220930T1800", seeds=[1, 2], n_members=3, n_leads=11)
    # A member that fails the RANGE detector ONLY: drive its t2m field out of the physical
    # interval while leaving its spectrum untouched, so the tail ratio stays quiet and
    # tail-only censoring keeps it alive. write_case's own `blowup` deliberately trips BOTH
    # detectors on the same member, which cannot tell the two rules apart.
    with h5py.File(os.path.join(d, "backward_fields_20220930T1800_seed001.h5"), "r+") as f:
        f["pred"][0, 1, 5:, CHANNELS.index("t2m"), :, :] = 400.0     # > range_hi = 350
    args = _fields_args(criterion="either", box=None, sensitivity_ceilings=[eh.TAIL_CEIL_DEFAULT])
    case = peh._load_case("pnw", os.path.join(d, "backward_fields_*_seed*.h5"),
                          os.path.join(d, "backward_spectrum_*_seed*.h5"), args, lambda m: None)
    sens = peh._survival_sensitivity(case, args, lambda m: None)
    # the range failure must be visible in BOTH, at the same lead
    assert case["stats"]["survival"][-1] < 1.0
    np.testing.assert_allclose(sens[eh.TAIL_CEIL_DEFAULT], case["stats"]["survival"])


def test_sensitivity_is_empty_under_the_range_criterion(tmp_path):
    """Varying a tail ceiling the headline curve never consults would be a curve about nothing."""
    d = str(tmp_path / "c")
    write_case(d, "20220930T1800", seeds=[1, 2], n_members=3, n_leads=9, blowup={(1, 0): 4})
    args = _fields_args(criterion="range", box=None)
    case = peh._load_case("pnw", os.path.join(d, "backward_fields_*_seed*.h5"),
                          os.path.join(d, "backward_spectrum_*_seed*.h5"), args, lambda m: None)
    assert peh._survival_sensitivity(case, args, lambda m: None) == {}


def test_nonfinite_variance_is_reported_and_excluded_not_zeroed(tmp_path):
    """A NaN grid point in a surviving member must not read as a variance of exactly zero.

    Under the tail criterion nothing inspects the field values, and NaN comparisons are
    False, so the range detector does not fire either: such a member stays alive and its
    deviations poison the accumulators. Zeroing them biases the spread -- and the e-folding
    fit taken from it -- silently low.
    """
    import h5py
    d = str(tmp_path / "c")
    write_case(d, "20220930T1800", seeds=[1, 2], n_members=3, n_leads=9)
    with h5py.File(os.path.join(d, "backward_fields_20220930T1800_seed001.h5"), "r+") as f:
        f["pred"][0, 0, 4:, 0, 3, 3] = np.nan     # one grid point, one member, deep leads
    lines = []
    args = _fields_args(criterion="tail", box=None)
    case = peh._load_case("pnw", os.path.join(d, "backward_fields_*_seed*.h5"),
                          os.path.join(d, "backward_spectrum_*_seed*.h5"), args, lines.append)
    assert case["spread_nonfinite_weight_fraction"] > 0
    assert any("non-finite ensemble variance" in m for m in lines), lines
    # the surviving cells still give a finite, positive spread rather than a NaN column
    assert np.isfinite(case["spread"][1:]).all()
    assert (case["spread"][1:] > 0).all()


def _case_box_args(**over):
    base = dict(box=PNW_BOX, case_box={})
    base.update(over)
    return argparse.Namespace(**base)


def test_case_box_defaults_to_the_shared_box():
    a = _case_box_args()
    assert peh._resolve_case_box("PNW heat dome", a) == PNW_BOX


def test_case_box_overrides_per_case():
    """The headline figure contrasts two regimes that are in different PLACES.

    One shared --box reduces a Gulf hurricane over a Pacific Northwest land box and labels
    the curve 'Hurricane Ian' -- a wrong number wearing the right name.
    """
    a = _case_box_args(case_box={"Hurricane Ian": "-90,-75,20,32"})
    assert peh._resolve_case_box("PNW heat dome", a) == PNW_BOX
    assert peh._resolve_case_box("Hurricane Ian", a) == (-90.0, -75.0, 20.0, 32.0)


def test_case_box_accepts_global_and_whitespace_separated():
    a = _case_box_args(case_box={"x": "global", "y": "-90 -75 20 32"})
    assert peh._resolve_case_box("x", a) is None
    assert peh._resolve_case_box("y", a) == (-90.0, -75.0, 20.0, 32.0)


@pytest.mark.parametrize("bad, match", [
    ("-90,-75,20", "4 numbers"),
    ("-90,-75,20,32,40", "4 numbers"),
    ("-90,-75,north,32", "not four numbers"),
    ("-90,-75,32,20", "south < north"),      # inverted
    ("-90,-75,-100,32", "south < north"),    # off the sphere
])
def test_case_box_rejects_malformed_boxes(bad, match):
    with pytest.raises(SystemExit, match=match):
        peh._resolve_case_box("x", _case_box_args(case_box={"x": bad}))


def test_per_case_boxes_reduce_over_different_domains(tmp_path):
    """End-to-end: the same dump reduced over two boxes must give two different means."""
    d = str(tmp_path / "c")
    write_case(d, "20220930T1800", seeds=[1], n_members=2, n_leads=7)
    g = os.path.join(d, "backward_fields_*_seed*.h5")
    sg = os.path.join(d, "backward_spectrum_*_seed*.h5")
    args = _fields_args(criterion="tail")
    north = peh._load_case("n", g, sg, args, lambda m: None, box=(-126.0, -110.0, 38.0, 55.0))
    south = peh._load_case("s", g, sg, args, lambda m: None, box=(-90.0, -75.0, 20.0, 32.0))
    assert north["box"] != south["box"]
    assert not np.allclose(north["truth_scalar"], south["truth_scalar"])


# ---------------------------------------------------------------------------
# Geopotential display units. ERA5 stores m^2/s^2; domain scientists read height in m,
# and so does the comparison paper we set numbers beside.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ch, geo", [("z", True), ("z500", True), ("z850", True),
                                     ("t2m", False), ("msl", False), ("tcwv", False),
                                     ("zebra", False), ("z500x", False)])
def test_is_geopotential_matches_the_level_naming(ch, geo):
    assert eh.is_geopotential(ch) is geo


def test_to_display_units_divides_by_g_and_powers_it():
    assert eh.to_display_units("z500", 49033.25) == pytest.approx(49033.25 / 9.80665)
    # a variance / power spectrum carries the square
    assert eh.to_display_units("z500", 100.0, power=2) == pytest.approx(100.0 / 9.80665 ** 2)
    # everything else passes straight through, at any power
    assert eh.to_display_units("msl", 101325.0) == 101325.0
    assert eh.to_display_units("msl", 101325.0, power=2) == 101325.0


def test_display_unit_tracks_the_conversion():
    assert eh.display_unit("z500") == "m" and eh.storage_unit("z500") == "m^2/s^2"
    assert eh.display_unit("t2m") == eh.storage_unit("t2m") == "K"
    assert eh.display_unit("nonesuch") == ""       # honest blank, never a guess


def test_geopotential_scalars_are_converted_and_the_ratio_is_not(tmp_path):
    """The dimensional curves move by g; the survival curve -- a ratio -- must not."""
    d = str(tmp_path / "c")
    write_case(d, "20210628T1200", seeds=[1, 2], n_members=3, n_leads=9,
               channels=["t2m", "z500"])
    g, sg = (os.path.join(d, "backward_fields_*_seed*.h5"),
             os.path.join(d, "backward_spectrum_*_seed*.h5"))
    conv = peh._load_case("c", g, sg, _fields_args(channel="z500", criterion="tail", box=None),
                          lambda m: None)
    raw = peh._load_case("c", g, sg, _fields_args(channel="z500", criterion="tail", box=None,
                                                  raw_geopotential=True), lambda m: None)
    np.testing.assert_allclose(raw["scalar"] / eh.DISPLAY_GRAVITY, conv["scalar"])
    np.testing.assert_allclose(raw["truth_scalar"] / eh.DISPLAY_GRAVITY, conv["truth_scalar"])
    np.testing.assert_allclose(raw["spread"] / eh.DISPLAY_GRAVITY, conv["spread"])
    np.testing.assert_allclose(raw["stats"]["survival"], conv["stats"]["survival"])


def test_a_non_geopotential_channel_is_untouched_by_the_convention(tmp_path):
    d = str(tmp_path / "c")
    write_case(d, "20210628T1200", seeds=[1, 2], n_members=3, n_leads=9)
    g, sg = (os.path.join(d, "backward_fields_*_seed*.h5"),
             os.path.join(d, "backward_spectrum_*_seed*.h5"))
    a = _fields_args(channel="t2m", criterion="tail", box=None)
    b = _fields_args(channel="t2m", criterion="tail", box=None, raw_geopotential=True)
    x = peh._load_case("c", g, sg, a, lambda m: None)
    y = peh._load_case("c", g, sg, b, lambda m: None)
    np.testing.assert_array_equal(x["scalar"], y["scalar"])


def test_the_efolding_rate_survives_the_conversion(tmp_path):
    """Only the intercept moves under a rescaling; the growth RATE is what we quote."""
    x = np.arange(8, dtype=float)
    y = np.exp(0.03 * x) * 17.0
    s_raw, i_raw, n_raw = eh.exponential_fit(x, y)
    s_cnv, i_cnv, n_cnv = eh.exponential_fit(x, y / eh.DISPLAY_GRAVITY)
    assert n_raw == n_cnv
    assert s_cnv == pytest.approx(s_raw)
    assert i_cnv == pytest.approx(i_raw - np.log(eh.DISPLAY_GRAVITY))
