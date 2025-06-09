import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    ambulance: bool
    frame: str | None
    confidence: float
    mode: Literal["simulation", "model"] = (
        "simulation" if os.getenv("ITCS_USE_DUMMY_MODEL") == "1" else "model"
    )
    frames_examined: int = 0


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
        # For tests we do not need to draw boxes; returning the frame is enough.
        return self._frame


class DummyModel:
    """
    Lightweight stand-in for the YOLO model, used in automated tests.
    This avoids loading large model weights and makes the service
    deterministic during testing.
    """

    def __init__(self):
        self.names = {0: "ambulance"}
        self.detect_ambulance = True

    def predict(self, source, conf: float, save: bool = False):
        if self.detect_ambulance:
            return [DummyResult(source, [DummyBox(0, max(conf, 0.9))])]
        return [DummyResult(source, [])]


if os.getenv("ITCS_USE_DUMMY_MODEL") == "1":
    logger.info("Using DummyModel for YOLO predictions (ITCS_USE_DUMMY_MODEL=1).")
    model = DummyModel()
else:
    # Load the YOLO model from weights folder once at startup for efficiency.
    logger.info("Loading YOLO model from weights/best.pt...")
    weights = Path(__file__).resolve().parent / "weights" / "best.pt"
    if not weights.is_file():
        raise RuntimeError(
            "Missing reviewed weights/best.pt; use ITCS_USE_DUMMY_MODEL=1 only for simulation"
        )
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if "ambulance" not in [str(name).lower() for name in model.names.values()]:
        raise RuntimeError("Model must explicitly label an ambulance class")
    logger.info("Model loaded successfully. Available classes: %s", model.names)


@app.get("/health")
async def health_check():
    return {"status": "ok", "mode": "simulation" if isinstance(model, DummyModel) else "model"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_video(file: Annotated[UploadFile, File()]):
    suffix = os.path.splitext(file.filename or "")[1]
    if suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(status_code=415, detail="Use a supported video filename")
    tmp_path = None
    cap = None

    try:
        # Persist the uploaded file to a temporary location for OpenCV.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            total = 0
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > 32 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="Video must be at most 32 MiB")
                tmp.write(chunk)

        # Open video and process frames
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=422, detail="Video could not be decoded")
        processed_frame = None
        frame_count = 0
        confidence_threshold = 0.75  # Increase threshold for better accuracy
        min_frames_with_ambulance = 1  # Only require one frame for demo
        ambulance_frames = 0
        highest_confidence = 0.0
        best_frame = None
        last_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame.shape[0] * frame.shape[1] > 3840 * 2160:
                raise HTTPException(status_code=413, detail="Frame exceeds 4K pixel limit")
            last_frame = frame.copy()

            # Run YOLO detection with the trained (or dummy) model
            results = model.predict(source=frame, conf=confidence_threshold, save=False)

            # Check if any detected object is an ambulance
            for result in results:
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        class_name = str(model.names.get(cls, "")).lower()
                        confidence = float(box.conf[0])

                        logger.debug("Detected %s with confidence %.4f", class_name, confidence)

                        if class_name == "ambulance" and confidence >= confidence_threshold:
                            ambulance_frames += 1
                            if confidence > highest_confidence:
                                highest_confidence = confidence
                                best_frame = result.plot()

                            if ambulance_frames >= min_frames_with_ambulance:
                                _, buffer = cv2.imencode(".jpg", best_frame)
                                frame_data = base64.b64encode(buffer).decode("utf-8")

                                return PredictionResponse(
                                    ambulance=True,
                                    frame=frame_data,
                                    confidence=highest_confidence,
                                    frames_examined=frame_count + 1,
                                )

            # Process at most 100 frames to avoid long processing times
            frame_count += 1
            if frame_count >= 100:
                break

            # Save the current frame as processed_frame for non-ambulance case
            if processed_frame is None and frame is not None:
                labeled_frame = frame.copy()
                cv2.putText(
                    labeled_frame,
                    "No Ambulance Detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                processed_frame = labeled_frame

        frame_data = None
        if processed_frame is not None:
            _, buffer = cv2.imencode(".jpg", processed_frame)
            frame_data = base64.b64encode(buffer).decode("utf-8")
        elif last_frame is not None:
            labeled_frame = last_frame.copy()
            cv2.putText(
                labeled_frame,
                "No Ambulance Detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            _, buffer = cv2.imencode(".jpg", labeled_frame)
            frame_data = base64.b64encode(buffer).decode("utf-8")

        if frame_count == 0:
            raise HTTPException(status_code=422, detail="Video contains no decodable frames")
        return PredictionResponse(
            ambulance=False, frame=frame_data, confidence=0.0, frames_examined=frame_count
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Video processing failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process video.",
        ) from exc
    finally:
        await file.close()
        if cap is not None:
            cap.release()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
