import asyncio
import base64
import importlib
import os
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from scripts.make_synthetic_clip import write_clip


def create_client():
    # Use the lightweight dummy model during tests to avoid loading YOLO weights.
    os.environ["ITCS_USE_DUMMY_MODEL"] = "1"
    module = importlib.import_module("main")
    return TestClient(module.app), module


class DummyCapture:
    def __init__(self, frames):
        self._frames = frames
        self._index = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self):
        self.released = True


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
    assert (
        client.post("/predict", files={"file": ("notes.txt", b"text", "text/plain")}).status_code
        == 415
    )


def test_empty_decoded_video_is_not_reported_as_negative(monkeypatch):
    client, module = create_client()
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda path: DummyCapture([]))
    assert (
        client.post("/predict", files={"file": ("empty.mp4", b"dummy", "video/mp4")}).status_code
        == 422
    )


def test_filename_does_not_force_detection_and_frame_limit_is_100(monkeypatch):
    client, module = create_client()
    module.model.detect_ambulance = False
    frames = _build_dummy_frames(110)
    capture = DummyCapture(frames)
    monkeypatch.setenv("ITCS_DEBUG_MODE", "true")
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda path: capture)
    result = client.post("/predict", files={"file": ("ambulance.mp4", b"dummy", "video/mp4")})
    assert result.status_code == 200
    assert result.json()["ambulance"] is False
    assert result.json()["frames_examined"] == 100
    assert capture._index == 100


def test_other_class_is_not_remapped_to_ambulance(monkeypatch):
    client, module = create_client()
    module.model.detect_ambulance = True
    monkeypatch.setattr(module.model, "names", {0: "person"})
    monkeypatch.setattr(
        module.cv2, "VideoCapture", lambda path: DummyCapture(_build_dummy_frames(1))
    )
    result = client.post("/predict", files={"file": ("sample.mp4", b"dummy", "video/mp4")})
    assert result.status_code == 200
    assert result.json()["ambulance"] is False


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

    monkeypatch.setattr(module.tempfile, "NamedTemporaryFile", track)
    result = client.post(
        "/predict", files={"file": ("large.mp4", b"x" * (32 * 1024 * 1024 + 1), "video/mp4")}
    )
    assert result.status_code == 413
    assert created and all(not Path(path).exists() for path in created)


def _post_video(client, name="clip.mp4"):
    return client.post("/predict", files={"file": (name, b"dummy", "video/mp4")})


def test_inference_runs_off_the_event_loop(monkeypatch):
    client, module = create_client()
    module.model.detect_ambulance = True
    loop_threads = []
    original_predict = module.model.predict

    def predict(**kwargs):
        try:
            asyncio.get_running_loop()
            loop_threads.append(threading.current_thread().name)
        except RuntimeError:
            pass
        return original_predict(**kwargs)

    monkeypatch.setattr(module.model, "predict", predict)
    monkeypatch.setattr(
        module.cv2, "VideoCapture", lambda path: DummyCapture(_build_dummy_frames(2))
    )
    assert _post_video(client).status_code == 200
    assert loop_threads == []


def test_detection_on_a_later_frame_reports_frames_examined(monkeypatch):
    client, module = create_client()
    module.model.detect_ambulance = False
    calls = []
    original_predict = module.model.predict

    def predict(**kwargs):
        calls.append(1)
        module.model.detect_ambulance = len(calls) == 3
        return original_predict(**kwargs)

    monkeypatch.setattr(module.model, "predict", predict)
    capture = DummyCapture(_build_dummy_frames(10))
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda path: capture)
    data = _post_video(client).json()
    assert data["ambulance"] is True
    assert data["frames_examined"] == 3
    assert capture._index == 3
    assert capture.released


def test_frame_larger_than_4k_is_rejected(monkeypatch):
    client, module = create_client()
    frames = [np.zeros((2161, 3841, 3), dtype=np.uint8)]
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda path: DummyCapture(frames))
    response = _post_video(client)
    assert response.status_code == 413
    assert response.json()["detail"] == "Frame exceeds 4K pixel limit"


def test_jpeg_encoding_failure_is_a_server_error(monkeypatch):
    client, module = create_client()
    module.model.detect_ambulance = True
    monkeypatch.setattr(
        module.cv2, "VideoCapture", lambda path: DummyCapture(_build_dummy_frames(1))
    )
    monkeypatch.setattr(module.cv2, "imencode", lambda ext, frame: (False, None))
    response = _post_video(client)
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to process video."


def test_mode_comes_from_the_loaded_model(monkeypatch):
    client, module = create_client()
    module.model.detect_ambulance = False
    monkeypatch.setattr(
        module.cv2, "VideoCapture", lambda path: DummyCapture(_build_dummy_frames(1))
    )
    assert _post_video(client).json()["mode"] == "simulation"
    monkeypatch.setattr(module, "current_mode", lambda: "model")
    assert _post_video(client).json()["mode"] == "model"


def test_real_decode_of_a_synthetic_clip(tmp_path):
    client, module = create_client()
    module.model.detect_ambulance = False
    clip = write_clip(tmp_path / "synthetic.avi", frames=7)
    with clip.open("rb") as handle:
        response = client.post(
            "/predict", files={"file": ("synthetic.avi", handle, "video/x-msvideo")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["ambulance"] is False
    assert data["frames_examined"] == 7
    preview = cv2.imdecode(
        np.frombuffer(base64.b64decode(data["frame"]), np.uint8), cv2.IMREAD_COLOR
    )
    assert preview.shape == (180, 320, 3)


def test_bytes_that_are_not_video_are_rejected_by_the_real_decoder():
    client, _ = create_client()
    response = client.post("/predict", files={"file": ("broken.mp4", b"not a video", "video/mp4")})
    assert response.status_code == 422


def test_committed_sample_clip_decodes():
    sample = Path(__file__).resolve().parent.parent / "samples" / "synthetic-traffic.avi"
    capture = cv2.VideoCapture(str(sample))
    try:
        frames = 0
        while capture.read()[0]:
            frames += 1
    finally:
        capture.release()
    assert frames == 12


def test_cors_origins_default_to_any(monkeypatch):
    client, module = create_client()
    monkeypatch.delenv("ITCS_CORS_ORIGINS", raising=False)
    assert module.allowed_origins() == ["*"]
    response = client.get("/health", headers={"Origin": "http://example.test"})
    assert response.headers["access-control-allow-origin"] == "*"


def test_cors_origins_can_be_restricted(monkeypatch):
    _, module = create_client()
    monkeypatch.setenv("ITCS_CORS_ORIGINS", "http://localhost:5173, https://ops.example.test,")
    assert module.allowed_origins() == ["http://localhost:5173", "https://ops.example.test"]
