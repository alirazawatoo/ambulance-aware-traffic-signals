import importlib
import os

import numpy as np
from fastapi.testclient import TestClient


def create_client():
    # Use the lightweight dummy model during tests to avoid loading YOLO weights.
    os.environ["ITCS_USE_DUMMY_MODEL"] = "1"
    module = importlib.import_module("main")
    return TestClient(module.app), module


class DummyCapture:
    def __init__(self, frames):
        self._frames = frames
        self._index = 0

    def isOpened(self):
        return True

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self):
        pass


def _build_dummy_frames(count: int = 5):
    return [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(count)]


def test_health_check():
    client, _ = create_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "simulation"}


def test_predict_ambulance_detected(monkeypatch, tmp_path):
    client, module = create_client()

    frames = _build_dummy_frames()

    def dummy_videocapture(path):
        return DummyCapture(frames)

    monkeypatch.setattr(module.cv2, "VideoCapture", dummy_videocapture)

    # Ensure the dummy model reports an ambulance.
    module.model.detect_ambulance = True

    video_file = tmp_path / "ambulance_video.mp4"
    video_file.write_bytes(b"dummy-video-content")

    with video_file.open("rb") as f:
        files = {"file": ("ambulance_video.mp4", f, "video/mp4")}
        response = client.post("/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["ambulance"] is True
    assert data["mode"] == "simulation"
    assert data["frames_examined"] == 1
    assert data["frame"] is not None
    assert data["confidence"] >= 0.75


def test_predict_no_ambulance_detected(monkeypatch, tmp_path):
    client, module = create_client()

    frames = _build_dummy_frames()

    def dummy_videocapture(path):
        return DummyCapture(frames)

    monkeypatch.setattr(module.cv2, "VideoCapture", dummy_videocapture)

    # Ensure the dummy model does not report an ambulance.
    module.model.detect_ambulance = False

    video_file = tmp_path / "traffic_video.mp4"
    video_file.write_bytes(b"dummy-video-content")

    with video_file.open("rb") as f:
        files = {"file": ("traffic_video.mp4", f, "video/mp4")}
        response = client.post("/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["ambulance"] is False
    assert data["frame"] is not None
    assert data["confidence"] == 0.0



def test_unsupported_extension_is_rejected():
    client, _ = create_client()
    assert client.post('/predict', files={'file': ('notes.txt', b'text', 'text/plain')}).status_code == 415


def test_empty_decoded_video_is_not_reported_as_negative(monkeypatch):
    client, module = create_client()
    monkeypatch.setattr(module.cv2, 'VideoCapture', lambda path: DummyCapture([]))
    assert client.post('/predict', files={'file': ('empty.mp4', b'dummy', 'video/mp4')}).status_code == 422


def test_filename_does_not_force_detection_and_frame_limit_is_100(monkeypatch):
    client, module = create_client()
    module.model.detect_ambulance = False
    frames = _build_dummy_frames(110)
    capture = DummyCapture(frames)
    monkeypatch.setenv('ITCS_DEBUG_MODE', 'true')
    monkeypatch.setattr(module.cv2, 'VideoCapture', lambda path: capture)
    result = client.post('/predict', files={'file': ('ambulance.mp4', b'dummy', 'video/mp4')})
    assert result.status_code == 200
    assert result.json()['ambulance'] is False
    assert result.json()['frames_examined'] == 100
    assert capture._index == 100


def test_other_class_is_not_remapped_to_ambulance(monkeypatch):
    client, module = create_client()
    module.model.detect_ambulance = True
    monkeypatch.setattr(module.model, 'names', {0: 'person'})
    monkeypatch.setattr(module.cv2, 'VideoCapture', lambda path: DummyCapture(_build_dummy_frames(1)))
    result = client.post('/predict', files={'file': ('sample.mp4', b'dummy', 'video/mp4')})
    assert result.status_code == 200
    assert result.json()['ambulance'] is False


def test_oversized_upload_is_rejected_and_temporary_file_removed(monkeypatch):
    client, module = create_client()
    import tempfile
    from pathlib import Path
    created = []
    original = tempfile.NamedTemporaryFile
    def track(*args, **kwargs):
        value = original(*args, **kwargs)
        created.append(value.name)
        return value
    monkeypatch.setattr(module.tempfile, 'NamedTemporaryFile', track)
    result = client.post('/predict', files={'file': ('large.mp4', b'x' * (32 * 1024 * 1024 + 1), 'video/mp4')})
    assert result.status_code == 413
    assert created and all(not Path(path).exists() for path in created)
