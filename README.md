# Ambulance-Aware Traffic Signals

A FastAPI computer-vision prototype that accepts a traffic video, scans up to 100 frames with a YOLO model, and returns whether an ambulance was detected. The response includes the highest confidence score and a base64-encoded annotated frame for use by a future traffic-management interface.

This repository demonstrates the detection and API boundary of an emergency-vehicle prioritization concept. It does **not** directly control traffic hardware.

## What is included

- `POST /predict` for multipart video uploads
- YOLO inference with a 0.75 confidence threshold
- A structured response containing the detection flag, confidence, and preview frame
- Temporary-file cleanup after each request
- A lightweight dummy model for deterministic API tests
- Positive and negative sample videos under `test_files/`

## Request flow

```text
Video upload
    |
    v
FastAPI /predict
    |
    v
OpenCV frame reader -- scans at most 100 frames
    |
    v
YOLO inference -- class 0 is mapped to "ambulance"
    |
    v
JSON result -- ambulance, confidence, annotated JPEG frame
```

## Repository layout

```text
.
├── smart-traffic-backend/
│   ├── main.py              # FastAPI service and inference pipeline
│   ├── requirements.txt     # Backend dependencies
│   └── tests/test_main.py   # API tests using the dummy model
├── test_files/              # Example traffic videos
└── public/fonts/            # Static font assets retained from the prototype
```

## API

### Health check

```http
GET /health
```

```json
{"status": "ok"}
```

### Analyze a video

```http
POST /predict
Content-Type: multipart/form-data
```

The multipart field is named `file`.

```json
{
  "ambulance": true,
  "frame": "<base64-encoded JPEG>",
  "confidence": 0.91
}
```

## Local setup

The service expects model weights at `smart-traffic-backend/weights/best.pt`. Those weights are not tracked in this repository, so a real-model run requires supplying a compatible YOLO checkpoint first.

```bash
cd smart-traffic-backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

For the deterministic test substitute instead of a model checkpoint:

```bash
cd smart-traffic-backend
ITCS_USE_DUMMY_MODEL=1 pytest
```

The tests cover the health endpoint plus positive and negative prediction responses with mocked video capture. They do not measure real-world model accuracy or validate traffic-signal hardware.

## Design notes

- The model is loaded once when the application starts.
- Uploaded videos are written to a temporary file because OpenCV consumes a file path.
- The service returns immediately after the first qualifying ambulance detection.
- `ITCS_DEBUG_MODE=true` enables filename-based demo behavior after ten frames. It should remain disabled in production-like evaluation.
- CORS is open in this prototype and should be restricted before deployment.

## Current boundaries

- The trained `weights/best.pt` checkpoint is not included.
- No frontend or traffic-light controller is tracked here.
- Upload size, media type, and authentication are not currently enforced.
- The bundled sample videos are demonstration inputs; their origin and reuse rights should be confirmed before public redistribution.

## Sensible next steps

1. Publish reproducible model provenance and evaluation metrics.
2. Add upload validation, request limits, and restricted CORS origins.
3. Move inference to a worker for longer videos and concurrent traffic.
4. Define a separate, fail-safe integration contract for any physical signal controller.
