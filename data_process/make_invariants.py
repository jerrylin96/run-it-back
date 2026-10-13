#!/usr/bin/env python3
# ============================================================================
"""Build the orography + land-sea-mask invariants for the 72-channel config.

Makani consumes orography and the land-sea mask as STATIC, input-only auxiliary
fields (never prediction targets). They are read at model-build time by
``makani/utils/auxiliary_fields.py``:

    get_orography(path)  -> reads HDF5 key "Z"   at [0, :, :], min/max -> [0, 1]
    get_land_mask(path)  -> reads HDF5 key "LSM" at [0, :, :], returned raw (0..1)

so the source arrays must be HDF5 with those exact keys and a leading singleton
axis: shape ``(1, nlat, nlon)``.

The WeatherBench2/ARCO -> makani converter (``convert_wb2_to_makani_input.py``)
handles only DYNAMIC, time-indexed channels — it has no invariant path — so the
static invariants must be produced separately. Upstream makani points at ECMWF
MARS / NCAR-RDA invariant files (``e5.oper.invariant.128_129_z`` /
``...128_172_lsm``); we instead derive them from the SAME ARCO zarr we convert
from. That zarr exposes ``geopotential_at_surface`` (= ERA5 surface geopotential,
param 129 — the orography proxy) and ``land_sea_mask`` (param 172) as static 2D
``(latitude, longitude)`` fields. They are the same physical ERA5 invariants on
the same 0.25 deg grid, so they are interchangeable with the CDS files:

  * Orography is RE-STANDARDIZED downstream ((x-mean)/std, preprocessor_helpers.py),
    which is invariant under any affine pre-transform — units, the /g factor, and
    the [0,1] min/max in get_orography all cancel. Only the spatial pattern (which
    ARCO and CDS share) matters. So we store the RAW geopotential under key "Z".
  * The land mask is NOT normalized (it is floor-binarized + one-hot encoded
    downstream), so we store the RAW fractional mask under key "LSM".

CRITICAL — grid orientation. The model's default grid runs latitude 90 -> -90
(North->South); the ARCO zarr is stored latitude-ascending. We align with the
exact same ``.sel(latitude=lat, longitude=lon)`` call the converter uses
(``coord_mode="match"``), so the invariants land on the identical grid/orientation
as the converted ``fields``. A mirrored invariant would feed the model a
hemisphere-flipped static cue — hence the geographic sanity checks below.

Examples
--------
# Build both invariants from the full-res ARCO zarr onto the 72-channel grid:
    python make_invariants.py \
        --input_file gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2 \
        --metadata_file examples/metadata_fcn3_72.json \
        --output_dir $MAKANI_DATA/invariants

# Validate the HDF5 file-format contract only (no network / zarr; needs h5py+numpy):
    python make_invariants.py --self_test --output_dir /tmp/inv_selftest

# Re-check ALREADY-WRITTEN invariants offline (geo-sanity gate; no network) — the
# confirmation train_fcn3.sbatch runs in its preflight:
    python make_invariants.py --check_existing \
        --metadata_file examples/metadata_fcn3_72.json \
        --output_dir $MAKANI_DATA/invariants
"""

import argparse
import os
import sys

import h5py as h5
import numpy as np

# the file-format contract that makani/utils/auxiliary_fields.py imposes
OROGRAPHY_KEY = "Z"
LANDMASK_KEY = "LSM"
# ARCO/WeatherBench2 variable names (verified against the store's .zmetadata)
ARCO_OROGRAPHY_VAR = "geopotential_at_surface"
ARCO_LANDMASK_VAR = "land_sea_mask"


def _write_invariant(path, key, field):
    """Write a single (nlat, nlon) field to HDF5 as key[0,:,:] with a leading singleton."""
    field = np.asarray(field, dtype=np.float32)
    if field.ndim != 2:
        raise ValueError(f"expected a 2D (lat, lon) field, got shape {field.shape}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5.File(path, "w") as f:
        f.create_dataset(key, data=field[None, :, :])  # shape (1, nlat, nlon)
    return field.shape


def _read_invariant(path, key):
    """Read back exactly as auxiliary_fields.py does: f[key][0, :, :]."""
    with h5.File(path, "r") as f:
        return f[key][0, :, :]


def _sanity_report(name, arr):
    arr = np.asarray(arr)
    a = arr.astype(np.float64)
    print(f"  {name}: shape={arr.shape} dtype={arr.dtype} min={a.min():.4g} max={a.max():.4g} mean={a.mean():.4g} nan={int(np.isnan(a).sum())}")


def geographic_gate(oro, lsm, latv, lonv):
    """HARD geographic sanity gate, shared by the ARCO build path and the offline
    --check_existing re-check. Both index the METADATA grid (= the grid the model
    assumes, 90->-90 / 0->359.75); match-alignment places the data on exactly that
    grid, so a flipped/mis-aligned grid (or the wrong variable) shows up as the
    Himalaya cell holding ocean, or the Sahara cell holding sea -> raise. Returns
    the land fraction on success; raises ValueError on failure."""
    latv = np.asarray(latv)
    lonv = np.asarray(lonv)

    def cell(la, lo):
        return int(np.abs(latv - la).argmin()), int(np.abs(lonv - (lo % 360)).argmin())

    iy_h, ix_h = cell(28.0, 87.0)  # Himalaya
    iy_p, ix_p = cell(0.0, 200.0)  # central Pacific (160W)
    iy_s, ix_s = cell(23.0, 13.0)  # Sahara
    oro_ok = oro[iy_h, ix_h] > oro[iy_p, ix_p]
    land_ok = lsm[iy_s, ix_s] > 0.5 and lsm[iy_p, ix_p] < 0.5
    land_frac = float(lsm.mean())
    print(f"  orography Himalaya(28N,87E)={oro[iy_h, ix_h]:.4g} vs open-Pacific(0N,160W)={oro[iy_p, ix_p]:.4g} " f"(expect Himalaya >> Pacific): {'OK' if oro_ok else 'FAIL'}")
    print(
        f"  landmask Sahara(23N,13E)={lsm[iy_s, ix_s]:.3g} (expect ~1) central-Pacific(0N,160W)={lsm[iy_p, ix_p]:.3g} (expect ~0) "
        f"land-fraction={land_frac:.3g} (expect ~0.25-0.35): {'OK' if land_ok else 'FAIL'}"
    )
    if not (oro_ok and land_ok):
        raise ValueError("geographic sanity FAILED — grid likely flipped/mis-aligned or wrong variable read; " "refusing (a hemisphere-flipped static cue is catastrophic).")
    if not (0.1 < land_frac < 0.5):
        # ERA5 fractional land fraction is ~0.33; warn widely (a narrow hard band would false-fail).
        print(f"  WARNING: land fraction {land_frac:.3g} outside the expected ~0.25-0.35; verify the mask.")
    return land_frac


def build_from_arco(input_file, metadata_file, output_dir, oro_name, lsm_name):
    import json
    import xarray as xr

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from makani.utils.dataloaders.wb2_helpers import gcs_storage_options

    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    lat = metadata["coords"]["lat"]
    lon = metadata["coords"]["lon"]
    nlat, nlon = len(lat), len(lon)
    print(f"target grid: {nlat} lat x {nlon} lon (lat[0]={lat[0]}, lat[-1]={lat[-1]})")

    storage_options = gcs_storage_options() if input_file.startswith(("gs://", "gcs://")) else {}
    ds = xr.open_dataset(input_file, engine="zarr", storage_options=storage_options)

    for var in (ARCO_OROGRAPHY_VAR, ARCO_LANDMASK_VAR):
        if var not in ds:
            raise KeyError(f"variable {var!r} not found in {input_file}. Available (first 40): {list(ds.data_vars)[:40]}")

    # Align to the metadata grid EXACTLY as convert_wb2_to_makani_input.py's default
    # (coord_mode="match"): xarray label-selection with the descending metadata lat list
    # reorders an ascending-stored zarr to 90->-90, so the invariants land on the SAME
    # grid/orientation as the converted fields. We support ONLY match: an invariant MUST be
    # on the model's grid, so a label mismatch should fail loudly here (KeyError) rather than
    # be forced through with a slice that the geographic gate below could only partly catch.
    ds = ds.sel(latitude=lat, longitude=lon)

    oro = np.asarray(ds[ARCO_OROGRAPHY_VAR].values, dtype=np.float32)  # raw surface geopotential
    lsm = np.asarray(ds[ARCO_LANDMASK_VAR].values, dtype=np.float32)  # raw fractional 0..1
    if oro.shape != (nlat, nlon) or lsm.shape != (nlat, nlon):
        raise ValueError(f"grid mismatch after align: oro {oro.shape}, lsm {lsm.shape}, expected {(nlat, nlon)}")

    # Hard-fail on NaN: get_orography's (x-min)/(max-min) and the downstream normalizer would
    # propagate a NaN across the whole static channel and silently poison training.
    for name, arr in (("orography", oro), ("land_sea_mask", lsm)):
        n_nan = int(np.isnan(arr).sum())
        if n_nan:
            raise ValueError(f"{name} has {n_nan} NaN(s) — refusing to write a NaN-bearing invariant")

    # ---- geographic sanity: a HARD gate, run BEFORE writing, so a flipped/mis-aligned grid
    # or the wrong variable can never be written and then trained on (shared with the offline
    # --check_existing re-check; see geographic_gate).
    print("sanity:")
    _sanity_report("orography(raw geopotential m^2/s^2)", oro)
    _sanity_report("land_sea_mask(raw fraction)", lsm)
    geographic_gate(oro, lsm, lat, lon)

    oro_path = os.path.join(output_dir, oro_name)
    lsm_path = os.path.join(output_dir, lsm_name)
    _write_invariant(oro_path, OROGRAPHY_KEY, oro)
    _write_invariant(lsm_path, LANDMASK_KEY, lsm)
    print(f"wrote {oro_path} (key {OROGRAPHY_KEY}) and {lsm_path} (key {LANDMASK_KEY})")

    # round-trip through the exact read contract
    assert _read_invariant(oro_path, OROGRAPHY_KEY).shape == (nlat, nlon)
    assert _read_invariant(lsm_path, LANDMASK_KEY).shape == (nlat, nlon)
    print("read-back via auxiliary_fields contract (f[key][0,:,:]): OK")


def check_existing(output_dir, metadata_file, oro_name, lsm_name):
    """Offline geo-sanity gate on ALREADY-WRITTEN invariant .h5 files (no network/zarr).

    Reads the two invariants exactly as makani/utils/auxiliary_fields.py does (f[key][0,:,:])
    and re-runs the SAME geographic gate build_from_arco runs before writing. Purpose: a
    seconds-long, GPU-free confirmation that the invariants a training job is about to consume
    are grid-aligned and finite — catching an invariant that was built by an older path, hand-
    edited, or partially written, BEFORE it silently poisons a multi-hour run with a hemisphere-
    flipped static cue. Wired into train_fcn3.sbatch's preflight. Raises (SystemExit/ValueError)
    on any problem; prints an OK line on success."""
    import json

    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    lat = metadata["coords"]["lat"]
    lon = metadata["coords"]["lon"]
    nlat, nlon = len(lat), len(lon)

    oro_path = os.path.join(output_dir, oro_name)
    lsm_path = os.path.join(output_dir, lsm_name)
    for path, key in ((oro_path, OROGRAPHY_KEY), (lsm_path, LANDMASK_KEY)):
        if not os.path.isfile(path):
            raise SystemExit(f"missing invariant {path} (key {key}) — run data_process/invariants.sbatch first")

    oro = np.asarray(_read_invariant(oro_path, OROGRAPHY_KEY), dtype=np.float32)
    lsm = np.asarray(_read_invariant(lsm_path, LANDMASK_KEY), dtype=np.float32)
    if oro.shape != (nlat, nlon) or lsm.shape != (nlat, nlon):
        raise SystemExit(f"grid mismatch: oro {oro.shape}, lsm {lsm.shape}, expected {(nlat, nlon)} from {metadata_file}")
    for name, arr in (("orography", oro), ("land_sea_mask", lsm)):
        n_nan = int(np.isnan(arr).sum())
        if n_nan:
            raise SystemExit(f"{name} in {output_dir} has {n_nan} NaN(s) — refusing to train on a NaN-bearing invariant")

    print(f"checking existing invariants in {output_dir} against grid {nlat} x {nlon} (lat[0]={lat[0]}, lat[-1]={lat[-1]}):")
    _sanity_report("orography(raw geopotential)", oro)
    _sanity_report("land_sea_mask(raw fraction)", lsm)
    geographic_gate(oro, lsm, lat, lon)
    print(f"INVARIANTS OK: {oro_name} + {lsm_name} pass the geographic sanity gate on the existing files.")


def self_test(output_dir):
    """Validate the HDF5 file-format contract with synthetic data — no network/zarr needed."""
    nlat, nlon = 721, 1440
    rng = np.random.default_rng(0)
    oro = rng.standard_normal((nlat, nlon)).astype(np.float32)
    lsm = (rng.random((nlat, nlon)) > 0.7).astype(np.float32)
    oro_path = os.path.join(output_dir, "orography.h5")
    lsm_path = os.path.join(output_dir, "land_sea_mask.h5")
    s1 = _write_invariant(oro_path, OROGRAPHY_KEY, oro)
    s2 = _write_invariant(lsm_path, LANDMASK_KEY, lsm)
    # read back exactly as makani does and assert byte-equality
    ro = _read_invariant(oro_path, OROGRAPHY_KEY)
    rl = _read_invariant(lsm_path, LANDMASK_KEY)
    assert s1 == (nlat, nlon) and s2 == (nlat, nlon), (s1, s2)
    assert ro.shape == (nlat, nlon) and rl.shape == (nlat, nlon)
    assert np.array_equal(ro, oro) and np.array_equal(rl, lsm)
    # mimic get_orography's min/max normalization to confirm the downstream reader path is sane
    norm = (ro - ro.min()) / (ro.max() - ro.min())
    assert 0.0 <= norm.min() and norm.max() <= 1.0
    print(f"self_test OK: keys {OROGRAPHY_KEY!r}/{LANDMASK_KEY!r}, shape (1,{nlat},{nlon}); " f"read-back f[key][0,:,:] matches; min/max normalization in [0,1].")


def main(args):
    if args.self_test:
        self_test(args.output_dir)
        return
    if args.check_existing:
        if not args.metadata_file:
            raise SystemExit("--metadata_file is required with --check_existing")
        check_existing(args.output_dir, args.metadata_file, args.orography_name, args.landmask_name)
        return
    if not args.input_file or not args.metadata_file:
        raise SystemExit("--input_file and --metadata_file are required unless --self_test / --check_existing is set")
    os.makedirs(args.output_dir, exist_ok=True)
    build_from_arco(args.input_file, args.metadata_file, args.output_dir, args.orography_name, args.landmask_name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_file", type=str, default=None, help="ARCO/WB2 zarr (gs://...) with geopotential_at_surface + land_sea_mask")
    p.add_argument("--metadata_file", type=str, default=None, help="metadata json defining the target lat/lon grid")
    p.add_argument("--output_dir", type=str, required=True, help="directory to write orography.h5 + land_sea_mask.h5")
    p.add_argument("--orography_name", type=str, default="orography.h5")
    p.add_argument("--landmask_name", type=str, default="land_sea_mask.h5")
    p.add_argument("--self_test", action="store_true", help="validate the HDF5 file-format contract with synthetic data only")
    p.add_argument(
        "--check_existing",
        action="store_true",
        help="offline geo-sanity gate on the EXISTING orography.h5/land_sea_mask.h5 in --output_dir "
        "(needs --metadata_file; no network) — used by train_fcn3.sbatch to confirm invariants before training",
    )
    main(p.parse_args())
