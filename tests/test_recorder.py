import csv

import cv2
import pytest

from follow.recorder import TRACK_COLUMNS, Recorder
from tests.test_follow_closed_loop import DT, FollowRig


def test_records_a_follow_flight(tmp_path):
    rig = FollowRig()
    rig.start()
    rec = Recorder(tmp_path / "run1", rig.clock())
    for _ in range(45):  # 1.5 s of frames
        rig.run(DT)
        frame = rig.scene.render()
        rec.add(rig.clock(), frame, frame, rig.follow, None, rig.nav.pose)
    rec.close()

    run = tmp_path / "run1"
    cap = cv2.VideoCapture(str(run / "raw.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 45
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (640, 360)
    assert (run / "overlay.mp4").stat().st_size > 0

    frames = list(csv.DictReader(open(run / "frames.csv")))
    assert [int(r["frame"]) for r in frames] == list(range(45))
    times = [float(r["t"]) for r in frames]
    assert times == sorted(times)

    rows = list(csv.DictReader(open(run / "track.csv")))
    assert list(rows[0].keys()) == TRACK_COLUMNS
    assert all(r["state"] == "LOCKED" and r["id"] == "1" for r in rows)
    for r in rows:
        assert 0 < float(r["cx"]) < 1 and 0 < float(r["h"]) <= 1
        assert 2.5 < float(r["distance_m"]) < 6
        assert r["drone_x"] != ""  # position tracking was on


def test_refuses_to_overwrite_a_recording(tmp_path):
    (tmp_path / "run1").mkdir()
    (tmp_path / "run1" / "raw.mp4").write_bytes(b"x")
    with pytest.raises(SystemExit):
        Recorder(tmp_path / "run1", 0.0)


def test_reuses_a_folder_that_never_got_video(tmp_path):
    run = tmp_path / "field1"
    Recorder(run, 0.0).close()  # drone video never arrived: just the CSV headers
    Recorder(run, 0.0).close()
