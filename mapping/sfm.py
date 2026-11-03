"""COLMAP structure-from-motion through pycolmap (pinned in mapping/requirements.txt).

Written against the pycolmap 4.2.1 API. Kept apart from the rest so the tests and --from-track never
need COLMAP installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SfmResult:
    points: np.ndarray  # (N, 3) in the reconstruction's own frame and units
    cams: dict[str, tuple[np.ndarray, np.ndarray]]  # image name -> (center, world_from_cam rotation)
    focal: float  # px, after bundle adjustment
    models: int  # how many separate pieces COLMAP split the flight into (we keep the biggest)


def run_sfm(work: Path, image_dir: Path, names: list[str], focal: float, w: int, h: int, seed: int = 0) -> SfmResult:
    try:
        import pycolmap
    except ImportError:
        raise SystemExit(
            "pycolmap isn't installed. See mapping/README.md: "
            "uv pip install --python .venv/bin/python -r mapping/requirements.txt"
        ) from None

    db = work / "database.db"
    if db.exists():
        db.unlink()  # COLMAP appends to an existing database; a rerun would match old frames
    sparse = work / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"  # one focal + one distortion term: all a toy lens can support
    reader.camera_params = f"{focal},{w / 2},{h / 2},0"  # start from the 60 degree FOV; BA refines it
    pycolmap.extract_features(db, image_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader)
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = 10  # ~3 s of keyframes either side; video, so no need for exhaustive pairs
    pycolmap.match_sequential(db, pairing_options=pairing)
    # COLMAP's defaults expect a good camera moving a lot. This one is 640x360, blurry, and often hovers, so
    # it starts from a narrower angle and fewer matches. Seeded, so the same flight maps the same way twice.
    opts = pycolmap.IncrementalPipelineOptions()
    opts.random_seed = seed
    opts.init_num_trials = 400
    opts.min_num_matches = 10
    opts.mapper.init_min_tri_angle = 4.0
    opts.mapper.init_min_num_inliers = 50
    opts.mapper.abs_pose_min_num_inliers = 15
    recs = pycolmap.incremental_mapping(db, image_dir, sparse, options=opts)
    recs = list(recs.values()) if isinstance(recs, dict) else list(recs)
    if not recs:
        raise SystemExit(
            "COLMAP couldn't build a model: too blurry, too little texture, or the drone only turned in place "
            "(rotation alone gives no depth). See mapping/README.md for how to fly for a map."
        )
    return read_model(max(recs, key=lambda r: r.num_reg_images()), len(recs))


def read_model(rec, models: int = 1) -> SfmResult:
    """A pycolmap Reconstruction (e.g. pycolmap.Reconstruction("<run>/sfm/sparse/0")) -> SfmResult."""
    cams = {}
    for img in rec.images.values():
        if img.has_pose:
            m = np.asarray(img.cam_from_world().matrix())  # [R | t], world -> camera
            cams[img.name] = (-m[:, :3].T @ m[:, 3], m[:, :3].T)
    pts = [p.xyz for p in rec.points3D.values() if p.track.length() >= 3 and p.error < 2.0]
    cam = next(iter(rec.cameras.values()))
    return SfmResult(np.asarray(pts, float).reshape(-1, 3), cams, float(cam.params[0]), models)
