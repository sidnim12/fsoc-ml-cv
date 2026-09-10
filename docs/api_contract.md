# API Contract For Unity Integration

This document is the stable contract between Unity, the FastAPI backend, and the ML/CV pipeline.

## Runtime Flow

```text
Unity virtual camera frame
        -> POST /predict-frame
        -> FastAPI decodes image with OpenCV
        -> SingleFramePipeline detects/classifies beacon candidates
        -> TargetTracker filters, predicts, and sets lock state
        -> FastAPI returns JSON
        -> Unity PID/gimbal uses control error or filtered position
```

Ground truth may be used for offline labels and evaluation only. It must not drive live gimbal control.

## Required Endpoints

### GET /health

Returns backend/model readiness.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "tracker_ready": true,
  "device": "cpu",
  "config": "configs/unity.yaml"
}
```

### POST /reset

Clears tracker state before a new sequence/run.

```json
{
  "status": "reset",
  "message": "Tracker state cleared"
}
```

### POST /predict-frame

Use `multipart/form-data`.

Fields:

```text
file: PNG/JPG frame
frame_index: optional integer
timestamp_s: optional float
session_id: optional string
```

Backend should decode the image as BGR OpenCV data, run the existing ML/CV pipeline, update the tracker, and return the response below.

## Stable Response

```json
{
  "target_found": true,
  "lock_state": "LOCKED",
  "filtered_x_px": 483,
  "filtered_y_px": 194,
  "predicted_x_px": 491,
  "predicted_y_px": 190,
  "control_error_x": 0.51,
  "control_error_y": 0.19,
  "confidence": 0.93
}
```

Recommended extra fields:

```json
{
  "frame_index": 42,
  "measurement_available": true,
  "candidate_count": 3,
  "temporally_verified": true,
  "using_prediction_only": false,
  "missed_frames": 0,
  "processing_time_ms": 18.4
}
```

## Backend Rules

- Load the model once at startup.
- Keep tracker state between frames.
- Call `/reset` before a new Unity run.
- Do not train inside the API.
- Do not implement PID inside the API.
- Do not use Unity ground truth for prediction.
- For multiple streams, keep one `TargetTracker` per `session_id`.

## Current ML/CV Assets

```text
configs/unity.yaml
src/pipeline.py
src/tracker.py
src/candidate_detector.py
src/beacon_classifier.py
models/checkpoints/official_1600x900_v2/best_classifier.pt
```

The checkpoint is ignored by Git and must be trained locally or shared separately.

## Official 1600x900 9k Update

The final 9000-frame Unity dataset has been inspected and trained on.

```text
Dataset root: data/raw/unity/official_1600x900/official_1600x900
Sequences: 30
Frames: 9000
Resolution: 1600x900
Checkpoint: models/checkpoints/official_1600x900_final/best_classifier.pt
Evaluation: outputs/unity-evaluation/official_1600x900_final_trained
Report: outputs/reports/official_1600x900_final_report.md
```

Current recommendation for backend integration: use the final checkpoint path above, but expect that confidence threshold tuning may still change before final demo.

## Current Best ML/CV Config

Use this config for backend integration testing:

```text
configs/unity_detector_positive.yaml
```

It points to:

```text
models/checkpoints/official_1600x900_detector_positive/best_classifier.pt
```

Expected offline full-frame metrics on the official 9000-frame Unity dataset:

```text
Candidate recall: 0.994643
Accepted detection recall: 0.994643
Filtered MAE: 2.950367 px
Locked frames: 99.000000%
FPS: 8.721445
```
