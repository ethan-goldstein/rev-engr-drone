# mapping/ — post-flight 3D map

Turns a recorded flight (`flights/<run>/` from `follow/recorder.py`) into the "security drone" map: a sparse
point cloud colored by height (turbo, like a LiDAR map), the drone's path in white and the followed
person's path in orange.

```bash
.venv/bin/python -m mapping.reconstruct flights/<run>/                # real footage: SfM on raw.mp4
.venv/bin/python -m mapping.reconstruct flights/<run>/ --from-track   # simulator flights: no SfM
```

It writes into the run folder:

| file | what |
|---|---|
| `map.html` | single-file interactive viewer (drag to orbit, scroll to zoom). Loads three.js 0.180.0 from cdnjs and OrbitControls from jsDelivr, both pinned and SRI-checked, so it needs internet the first time. |
| `map.mp4` | a 10 s fly-around of the same map, rendered with OpenCV (`--no-video` skips it) |
| `sfm/` | SfM working files: `images/` (the keyframes), `database.db`, `sparse/` (COLMAP model). Rebuilt on every run. |

Options: `--window 0.333` sets the keyframe spacing in seconds.

## Showreel: tracking replay + 3D map, side by side

```bash
.venv/bin/python -m mapping.reel flights/field4 flights/field3      # -> video/reels/<run>.mp4
```

For each flight:
- **Replay.** Follow mode runs offline over the drone video, as if F had been pressed at the start. The
  lock box, FACE LOCK card and telemetry graphs go into `<run>/replay/`. Night footage is brightened first.
- **Map.** COLMAP maps the replay, with looser settings for this toy camera and a fixed seed. It tries a
  couple of keyframe spacings and keeps whichever maps the most frames.
- **Compose.** The map orbits slowly, and the drone and person paths draw in as the flight plays.

The finished map is cached in `<run>/replay/map.npz`, along with `map.html`. Delete `map.npz` to redo it.
If a flight can't be mapped (too dark, or the drone barely moved), the map panel says why.

## Installing COLMAP (SfM only)

`--from-track` and the tests need nothing new. The SfM path needs `pycolmap`, pinned in
`mapping/requirements.txt`:

```bash
uv pip install --python .venv/bin/python -r mapping/requirements.txt
```

`pycolmap==4.2.1` ships a wheel for this setup: CPython 3.12 on macOS arm64, built for macOS 14+. That's about
20 MB, and its only dependency is numpy, which is already here. Don't install the `[panorama]` extra:
it pulls in `opencv-python-headless`, which conflicts with our `opencv-contrib-python`.

**Fallback if the wheel won't install** (macOS older than 14, or a future Python with no wheel yet):
`brew install colmap` gives the COLMAP command-line tools and GUI. You can run them on the keyframes this
tool saves, but this tool can't read the result yet. That needs a small reader for COLMAP's text model:

```bash
colmap feature_extractor --database_path flights/<run>/sfm/database.db --image_path flights/<run>/sfm/images \
  --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL
colmap sequential_matcher --database_path flights/<run>/sfm/database.db
colmap mapper --database_path flights/<run>/sfm/database.db --image_path flights/<run>/sfm/images \
  --output_path flights/<run>/sfm/sparse
colmap gui   # File > Import model: flights/<run>/sfm/sparse/0
```

## How it works

1. **Keyframes.** Score every frame by the variance of its Laplacian, and keep the sharpest one in each
   ~1/3 s window. Motion blur and JPEG mush score low.
2. **SfM.** COLMAP via pycolmap, with SIFT features and sequential matching (10 neighbours either side).
   It uses one shared camera (SIMPLE_RADIAL), starting from the 60° FOV focal length; bundle adjustment
   refines it. If COLMAP splits the flight into pieces, the map shows the biggest.
3. **Up and scale.**
   - *Up,* best cue first:
     - The ground: the biggest flat patch of points, below every camera and within 35° of the camera's
       own up (RANSAC). This ignores walls.
     - With no ground, the plane of the flight path, because the drone flies roughly level. A slow
       altitude drift tilts this, so it comes second.
     - The camera's own up axis. This is off by however much the camera looks down.
   - *Scale:* place the person at `distance_m` along each LOCKED frame's box-center ray, and pick the
     scale at which their middle sits 0.85 m above the reconstructed ground. That's half the 1.7 m the
     follow controller assumes. It works whether or not COLMAP kept the person, and it drops anyone
     who moves.
     - With no ground plane, it falls back to the brief's literal method. The depth of the points in
       the middle of the box is "how far the person is" in SfM units, matched median-to-median with
       `distance_m`. That's only right if the person held still. If they walked, those points are the
       background behind them: on a synthetic walking flight it made the map 29% too small.
   - The map is then rotated so the first frame faces +y (the drone's heading 0), with the ground at
     z = 0.
4. **Height colors.** Turbo from the 2nd to the 98th percentile of height, so a few stray points don't
   flatten everything.
5. **Person path.** Along each frame's box-center ray, at `distance_m` deep. That is pinhole depth,
   which is how the follow controller computes it. Frames that are LOST, have no box, or whose box
   touches the top or bottom edge (a cut-off person reads too far) are skipped. The path is smoothed
   over about 1 s.

`--from-track` skips 1–3. The drone path comes from `drone_x/drone_y` at an assumed 1.2 m height. The
person is placed from the drone heading plus the box ray. On a 30 s simulator flight it lands within
3 cm (median) of where the sim person really was.

## What quality to expect

**Measured on synthetic footage.** No real F19 flight has been recorded yet. The test flights are
20 s, 300 frames, 640x360 at 15 fps:
- a textured room, plus a rigid textured box standing in for the person;
- JPEG quality 55, 40% of frames motion-blurred;
- a drone arc around the person at 1.3 m.

| | person stands still | person walks |
|---|---|---|
| keyframes registered | 60 / 60 | 60 / 60 |
| points | 12,014 | 11,532 |
| map size vs true | 0.99× | 0.99× |
| up error | 0.4° | 1.1° |
| drone path error | 3 cm median | 3 cm median |
| person path error | 0.23 m median | 0.20 m median |

The timed run (person standing still) took 21 s end to end on this Mac's CPU. That's a best case: flat ground, a scene covered in texture, and a "person" exactly 1.7 m tall. Real
footage will be worse:

- **Sparse, not a scan.** Edges, textured ground and trees give points; sky, blank walls, water and
  glare give nothing. The viral-post look comes from height coloring on a dark background, not from
  density.
- **It needs sideways motion.** Turning in place gives COLMAP no parallax, so frames won't register or
  depths will be garbage. Following someone straight ahead gives weak parallax. Slow arcs or sidesteps
  around the person make the best maps (that's what the synthetic flights do).
- **Blur and compression cost frames.** Expect some keyframes to stay unregistered (the viewer shows
  "N of M registered"), and sometimes a split into pieces.
- **Scale: think ±10%.** `distance_m` assumes a 1.7 m person, so someone 1.6 m tall shifts it by a few
  %. Sloped ground or a drone flying very low (under about 0.5 m) makes it worse. The viewer says which
  method set the scale, and says "SfM units" if neither could.
- **The person path** is only as good as the box-size distance: about 0.2 m in the synthetic runs, and
  more for a real detector box.
## Tests

```bash
.venv/bin/python -m pytest tests/test_reconstruct.py
```

The tests cover keyframe picking, height coloring, up/ground/scale/alignment, person placement, the
HTML and video writers, and the `--from-track` CLI. They use a synthetic world pushed through a random
similarity transform, so COLMAP isn't needed. `mapping/sfm.py` (the pycolmap calls) isn't unit-tested;
it was exercised by the synthetic-footage runs above.
