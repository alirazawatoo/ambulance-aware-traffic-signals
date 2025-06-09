import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_FRAMES = 100
MAX_FRAME_PIXELS = 3840 * 2160
CONFIDENCE_THRESHOLD = 0.75
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "best.pt"


def allowed_origins() -> list[str]:
    """Comma-separated ITCS_CORS_ORIGINS; "*" (the default) allows any origin."""
    raw = os.getenv("ITCS_CORS_ORIGINS", "*")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(title="Ambulance priority video API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    ambulance: bool
    frame: str | None
    confidence: float
    mode: Literal["simulation", "model"]
    frames_examined: int


class DummyBox:
    def __init__(self, cls_id: int, confidence: float):
        # These mimic the minimal attributes accessed from YOLO boxes.
        self.cls = [cls_id]
        self.conf = [confidence]


class DummyResult:
    def __init__(self, frame, boxes):
        self._frame = frame
        self.boxes = boxes

    def plot(self):
        return self._frame


class DummyModel:
    """
    Stand-in for the YOLO model in simulation mode and tests: no weights are
    loaded, and it reports an ambulance on every frame while detect_ambulance
    is set, regardless of the video content.
    """

    def __init__(self):
        self.names = {0: "ambulance"}
        self.detect_ambulance = True

    def predict(self, source, conf: float, save: bool = False):
        if self.detect_ambulance:
            return [DummyResult(source, [DummyBox(0, max(conf, 0.9))])]
        return [DummyResult(source, [])]


def load_model():
    if os.getenv("ITCS_USE_DUMMY_MODEL") == "1":
        logger.info("Using DummyModel for YOLO predictions (ITCS_USE_DUMMY_MODEL=1).")
        return DummyModel()
    if not WEIGHTS_PATH.is_file():
        raise RuntimeError(
            "Missing reviewed weights/best.pt; use ITCS_USE_DUMMY_MODEL=1 only for simulation"
        )
    from ultralytics import YOLO

    yolo = YOLO(str(WEIGHTS_PATH))
    if "ambulance" not in [str(name).lower() for name in yolo.names.values()]:
        raise RuntimeError("Model must explicitly label an ambulance class")
    logger.info("Model loaded. Available classes: %s", yolo.names)
    return yolo


model = load_model()


def current_mode() -> Literal["simulation", "model"]:
    return "simulation" if isinstance(model, DummyModel) else "model"


def encode_jpeg(frame) -> str:
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buffer).decode("ascii")


def label_no_ambulance(frame):
    labeled = frame.copy()
    cv2.putText(
        labeled, "No Ambulance Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )
    return labeled


def strongest_ambulance(results):
    """Return (confidence, result) for the most confident qualifying ambulance box, or None."""
    best = None
    for result in results:
        for box in result.boxes:
            class_name = str(model.names.get(int(box.cls[0]), "")).lower()
            confidence = float(box.conf[0])
            logger.debug("Detected %s with confidence %.4f", class_name, confidence)
            if class_name != "ambulance" or confidence < CONFIDENCE_THRESHOLD:
                continue
            if best is None or confidence > best[0]:
                best = (confidence, result)
    return best


def scan_video(path: str) -> PredictionResponse:
    """Decode up to MAX_FRAMES frames and stop at the first qualifying ambulance.

    Decoding and inference block, so the endpoint runs this on a worker thread.
    """
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise HTTPException(status_code=422, detail="Video could not be decoded")
        first_frame = None
        frames_examined = 0
        while frames_examined < MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[0] * frame.shape[1] > MAX_FRAME_PIXELS:
                raise HTTPException(status_code=413, detail="Frame exceeds 4K pixel limit")
            frames_examined += 1
            if first_frame is None:
                first_frame = frame

            results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, save=False)
            hit = strongest_ambulance(results)
            if hit is not None:
                confidence, result = hit
                return PredictionResponse(
                    ambulance=True,
                    frame=encode_jpeg(result.plot()),
                    confidence=confidence,
                    mode=current_mode(),
                    frames_examined=frames_examined,
                )

        if first_frame is None:
            raise HTTPException(status_code=422, detail="Video contains no decodable frames")
        return PredictionResponse(
            ambulance=False,
            frame=encode_jpeg(label_no_ambulance(first_frame)),
            confidence=0.0,
            mode=current_mode(),
            frames_examined=frames_examined,
        )
    finally:
        cap.release()


@app.get("/health")
async def health_check():
    return {"status": "ok", "mode": current_mode()}


@app.post("/predict", response_model=PredictionResponse)
async def predict_video(file: Annotated[UploadFile, File()]):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Use a supported video filename")
    tmp_path = None
    try:
        # OpenCV needs a real file path, so the upload is spooled to disk with a size cap.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            total = 0
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video must be at most 32 MiB")
                tmp.write(chunk)
        return await run_in_threadpool(scan_video, tmp_path)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Video processing failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process video.",
        ) from exc
    finally:
        await file.close()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
