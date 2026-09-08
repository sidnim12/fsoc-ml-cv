# FSOC ML/CV Layer - PS 26169

This repository contains the ML/CV layer for a Unity-based Free-Space Optical Communication virtual test environment.

Free-Space Optical Communication uses a narrow laser beam to transmit data between moving terminals such as satellites, UAVs, or ground stations. Because the beam is extremely narrow, the system must find the remote terminal, detect its optical beacon, align the camera or gimbal, keep the target centred while both terminals move, and recover if tracking is lost.

The current project builds and validates the computer-vision and machine-learning side of that loop using synthetic virtual-camera data.

## System Goal

The complete prototype should demonstrate:

```text
Search -> detect -> identify -> align -> track -> disturb -> lose lock -> reacquire
```

The planned closed-loop architecture is:

```text
Unity creates camera frame
  -> OpenCV preprocessing
  -> bright candidate detection
  -> CNN beacon classification
  -> temporal verification
  -> Kalman prediction
  -> PID pan/tilt command
  -> Unity rotates virtual gimbal
```

The ML/CV layer will eventually return PID-ready data like:

```json
{
  "target_found": true,
  "target_id": "Terminal_B",
  "x_px": 486,
  "y_px": 192,
  "predicted_x_px": 493,
  "predicted_y_px": 190,
  "confidence": 0.96
}
```

## Work Completed So Far

### Phase 1: Synthetic Dataset Generation

Implemented in:

```text
src/generate_dataset.py
```

The generator creates 500 synthetic RGB images by default at 640 x 480 resolution.

Generated scenario distribution:

```text
S1 clean_target: 100
S2 target_with_stars: 100
S3 varied_target_size_brightness: 100
S4 sensor_noise_and_blur: 75
S5 target_with_false_beacons: 75
S6 target_not_visible: 50
```

Outputs:

```text
data/raw/python-generated/
data/labels/labels.csv
outputs/dataset-preview/sample_grid.png
```

Command used:

```powershell
.\.venv\Scripts\python.exe src\generate_dataset.py --num-images 500 --width 640 --height 480 --seed 42 --overwrite
```

### Phase 2: Image Preprocessing

Implemented in:

```text
src/preprocessing.py
```

Processing flow:

```text
Original BGR image
  -> grayscale
  -> optional CLAHE contrast normalization
  -> Gaussian blur
  -> fixed, Otsu, or adaptive thresholding
  -> morphological opening and closing
  -> clean binary image
```

Final preprocessing output count:

```text
Processed images: 500
Failed images: 0
Binary PNG count: 500
```

Best thresholding method for the current synthetic dataset:

```text
fixed threshold at 180 for candidate detection
```

Earlier comparison:

```text
fixed threshold 200: center hit rate 448/450, low clutter
otsu: center hit rate 450/450, much higher clutter
```

Outputs:

```text
outputs/preprocessing/binary/
outputs/preprocessing/comparisons/
```

### Phase 3: Candidate Detection

Implemented in:

```text
src/candidate_detector.py
```

The detector reuses the preprocessing module, finds external contours, extracts candidate measurements, filters invalid objects, ranks candidates with a non-ML baseline score, saves overlays, and evaluates candidate recall using ground truth.

Final command used:

```powershell
.\.venv\Scripts\python.exe src\candidate_detector.py --input data\raw\python-generated --output outputs\candidate-detection --labels data\labels\labels.csv --threshold-method fixed --threshold 180 --blur-kernel 5 --morph-kernel 3 --max-overlays 30
```

Final metrics:

```text
Processed images: 500
Failed images: 0
Candidate CSV rows: 662
Overlays saved: 30
Candidate recall: 1.0000
Missed visible targets: 0
Average candidates per frame: 1.324
Frames with zero candidates: 10
Baseline top-1 accuracy: 0.9778
Target-absent frames with false detections: 40 / 50
```

Multiple-beacon verification:

```text
Scenario 5 frames: 75
Frames with 2+ candidates: 66
Maximum detected candidates in one frame: 6
Average candidates per multiple-beacon frame: 2.987
```

Outputs:

```text
outputs/candidate-detection/candidates.csv
outputs/candidate-detection/summary.json
outputs/candidate-detection/overlays/
outputs/candidate-detection/overlay_montage_20.png
```

### Phase 4: CNN Patch Preparation

Implemented in:

```text
src/prepare_patches.py
```

This stage converts detected candidates into 32 x 32 RGB patches for CNN training.

Labels are assigned only from ground truth and distance to the real target:

```text
correct: visible target and candidate is within 12 px of ground truth
false: stars, decoys, noise, or target-absent candidates
```

Final patch dataset:

```text
Total patches: 662
Correct patches: 450
False patches: 212
Positive-to-negative ratio: 2.122642
```

Split counts:

```text
train:      correct 314, false 145
validation: correct 67,  false 34
test:       correct 69,  false 33
```

Outputs:

```text
data/processed/train/correct/
data/processed/train/false/
data/processed/validation/correct/
data/processed/validation/false/
data/processed/test/correct/
data/processed/test/false/
data/processed/patch_labels.csv
data/processed/dataset_summary.json
outputs/patch-preview/correct_grid.png
outputs/patch-preview/false_grid.png
```

Command used:

```powershell
.\.venv\Scripts\python.exe src\prepare_patches.py --images data\raw\python-generated --labels data\labels\labels.csv --candidates outputs\candidate-detection\candidates.csv --output data\processed --patch-size 32 --crop-size 40 --match-tolerance 12 --seed 42 --max-negative-ratio 3.0 --overwrite
```

### Phase 5: CNN Definition And Training

Implemented in:

```text
src/beacon_classifier.py
src/train_classifier.py
```

The CNN classifies each 32 x 32 RGB candidate patch as:

```text
Class 0: false
Class 1: correct
```

Model architecture:

```text
Conv 3 -> 16 + BatchNorm + ReLU + MaxPool
Conv 16 -> 32 + BatchNorm + ReLU + MaxPool
Conv 32 -> 64 + BatchNorm + ReLU
AdaptiveAvgPool 1 x 1
Dropout 0.3
Linear 64 -> 2 logits
```

Training details:

```text
Loss: CrossEntropyLoss
Optimizer: AdamW
Learning rate: 0.001
Weight decay: 0.0001
Batch size: 32
Max epochs: 30
Early stopping patience: 6
Selected device: CPU
Trainable parameters: 23,938
Early stopping epoch: 14
```

Final test metrics:

```text
Accuracy:  0.8039
Precision: 0.9016
Recall:    0.7971
F1-score:  0.8462
ROC-AUC:   0.8915
```

Confusion matrix:

```text
True false predicted false:    27
True false predicted correct:   6
True correct predicted false:  14
True correct predicted correct: 55
```

Outputs:

```text
models/checkpoints/best_classifier.pt
models/checkpoints/last_classifier.pt
outputs/training/history.csv
outputs/training/training_curves.png
outputs/training/confusion_matrix.png
outputs/training/classification_report.json
outputs/training/test_metrics.json
outputs/training/sample_predictions.png
outputs/training/inference_smoke_test.json
```

Training command:

```powershell
.\.venv\Scripts\python.exe src\train_classifier.py --metadata data\processed\patch_labels.csv --data-root data\processed --output-dir outputs\training --checkpoint-dir models\checkpoints --epochs 30 --batch-size 32 --learning-rate 0.001 --weight-decay 0.0001 --patience 6 --seed 42 --num-workers 0 --device auto
```

Note: PyTorch is installed in the project virtual environment as `torch 2.14.0+cpu`.

### Phase 6: Complete Single-Frame Inference Pipeline

Implemented in:

```text
src/pipeline.py
```

This phase connects the completed ML/CV pieces for one independent camera frame:

```text
frame or image path
  -> input validation and BGR conversion
  -> preprocessing
  -> bright candidate detection
  -> candidate patch extraction
  -> CNN classification
  -> CNN/CV fused ranking
  -> PID-ready coordinate output
```

This phase does not implement temporal verification, Kalman tracking, PID control, FastAPI, or Unity communication yet.

Pipeline outputs:

```text
outputs/pipeline-test/result.json
outputs/pipeline-test/annotated_result.png
```

Command used:

```powershell
.\.venv\Scripts\python.exe src\pipeline.py --image data\raw\python-generated\synthetic_0378_target_with_false_beacons.png --config configs\default.yaml --checkpoint models\checkpoints\best_classifier.pt --output outputs\pipeline-test --device cpu
```

Smoke-test result on the multiple-beacon frame:

```text
Target found: true
Status: target_detected
Candidates kept: 4
Selected candidate ID: 2
CNN probability: 0.7920
CV baseline score: 0.7225
Fused score: 0.7781
Pixel error: dx=238.0 px, dy=204.0 px
Control error: pan=0.74375, tilt=-0.85
```

The pipeline keeps every valid bright candidate and records per-candidate CNN probability, false probability, predicted class, CNN confidence, CV baseline score, and fused score.

Fusion formula:

```text
fused_score = 0.80 * correct_probability + 0.20 * baseline_score
```

The weights are configurable in `configs/default.yaml` and must sum to `1.0`.

Coordinate sign convention:

```text
image_error_x_px = target_x - frame_center_x
image_error_y_px = target_y - frame_center_y
```

Image-space meaning:

```text
Right = positive X
Left = negative X
Down = positive Y
Up = negative Y
```

Control-space meaning:

```text
control_error_x = image_error_x_px / frame_center_x
control_error_y = -image_error_y_px / frame_center_y
```

So right is positive pan, left is negative pan, up is positive tilt, and down is negative tilt. Normalized control errors are clamped to `[-1, 1]`.

Phase 6 tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline.py -v
```

Result:

```text
10 passed
```

### Phase 7: Temporal Verification And Kalman Tracking

Implemented in:

```text
src/temporal_verifier.py
src/tracker.py
tests/test_temporal_verifier.py
tests/test_tracker.py
pytest.ini
```

This phase adds memory across ordered frames. The single-frame pipeline still detects and ranks candidates, while the new tracking layer decides whether the same beacon is persisting over time, predicts where it should appear next, and keeps a lock state for downstream control.

Tracking flow:

```text
ordered frame sequence
  -> Phase 6 single-frame candidate ranking
  -> candidate association near Kalman prediction
  -> temporal verification
  -> Kalman correction or prediction-only coast
  -> lock-state update
  -> tracking CSV, summary JSON, annotated MP4
```

Temporal verification:

```text
Default mode: persistence
Default history window: 5 frames
Default confirmations needed: 3
Blink mode: implemented but disabled by default
```

The temporal verifier returns:

```text
temporally_verified
confirmation_count
history_size
verification_mode
blink_match
```

Kalman tracker:

```text
State vector:      [x, y, vx, vy]
Measurement:       [x, y]
Motion model:      constant velocity
Implementation:    NumPy only
```

Candidate association uses the predicted position from the Kalman filter and rejects candidates outside the gate:

```text
Default association gate: 60 px
association_score = 0.40 * fused_score + 0.60 * position_score
```

Lock states:

```text
SEARCHING  - no usable track yet
ACQUIRING  - candidates are being confirmed over time
LOCKED     - target is temporally verified and measured
COASTING   - target is temporarily missing, using prediction only
LOST       - too many missed frames, tracker must reacquire
```

Phase 7 smoke test:

```powershell
.\.venv\Scripts\python.exe src\tracker.py --image-dir outputs\tracking-smoke-input --config configs\default.yaml --checkpoint models\checkpoints\best_classifier.pt --output outputs\tracking-test --device cpu --fps 12
```

Smoke-test result:

```text
Total frames: 18
Measurements: 16
Locked frames: 14
Coasting frames: 2
Lost frames: 0
Maximum consecutive missed frames: 2
Mean association distance: 1.109044 px
```

During the smoke test, frames 10 and 11 intentionally hide the target. The tracker switches to `COASTING` with `using_prediction_only=true`, then returns to `LOCKED` on frame 12.

Outputs:

```text
outputs/tracking-test/tracking_results.csv
outputs/tracking-test/tracking_summary.json
outputs/tracking-test/annotated_tracking.mp4
```

Phase 7 tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_temporal_verifier.py tests\test_tracker.py -v
.\.venv\Scripts\python.exe -m pytest -v
```

Result:

```text
23 targeted Phase 7 tests passed
33 total project tests passed
```

## Current Project Structure

```text
fsoc-ml-cv/
  configs/
    default.yaml
  data/
    labels/
      labels.csv
    raw/
      python-generated/
      unity/
    processed/
      train/
        correct/
        false/
      validation/
        correct/
        false/
      test/
        correct/
        false/
      patch_labels.csv
      dataset_summary.json
  models/
    checkpoints/
      best_classifier.pt
      last_classifier.pt
    exported/
  notebooks/
    01_image_basics.ipynb
    02_beacon_detection.ipynb
    03_dataset_analysis.ipynb
  outputs/
    candidate-detection/
    dataset-preview/
    patch-preview/
    preprocessing/
    pipeline-test/
    tracking-test/
    training/
  review/
    progress_2026-09-07.md
    progress_2026-09-08.md
  src/
    generate_dataset.py
    preprocessing.py
    candidate_detector.py
    prepare_patches.py
    beacon_classifier.py
    train_classifier.py
    temporal_verifier.py
    tracker.py
    pipeline.py
    evaluate.py
    api.py
  tests/
    test_detector.py
    test_pipeline.py
    test_temporal_verifier.py
    test_tracker.py
  pytest.ini
```

## Notebooks

The notebooks are intended for explanation, inspection, and presentation:

```text
notebooks/01_image_basics.ipynb
```

Explores generated images, pixel intensity distributions, and preprocessing threshold methods.

```text
notebooks/02_beacon_detection.ipynb
```

Explores candidate detection outputs, candidate counts by scenario, multiple-beacon verification, overlays, and recall metrics.

```text
notebooks/03_dataset_analysis.ipynb
```

Explores patch dataset balance, split counts, patch previews, training history, test metrics, training artifacts, the Phase 6 single-frame inference output, and the Phase 7 tracking smoke summary.

## Setup

Install requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verify PyTorch:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Run notebooks:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

## Important Output Locations

Dataset:

```text
data/raw/python-generated/
data/labels/labels.csv
outputs/dataset-preview/sample_grid.png
```

Preprocessing:

```text
outputs/preprocessing/binary/
outputs/preprocessing/comparisons/
```

Candidate detection:

```text
outputs/candidate-detection/candidates.csv
outputs/candidate-detection/summary.json
outputs/candidate-detection/overlays/
```

Patch dataset:

```text
data/processed/patch_labels.csv
data/processed/dataset_summary.json
outputs/patch-preview/
```

CNN training:

```text
models/checkpoints/
outputs/training/
```

Single-frame pipeline:

```text
outputs/pipeline-test/result.json
outputs/pipeline-test/annotated_result.png
```

Sequence tracking:

```text
outputs/tracking-test/tracking_results.csv
outputs/tracking-test/tracking_summary.json
outputs/tracking-test/annotated_tracking.mp4
```

## Future Plan

### Next Phase: End-To-End Sequence Pipeline

Extend the current single-frame pipeline with temporal verification and Kalman prediction.

Planned file:

```text
src/pipeline.py
```

Expected additions:

```text
- A reusable sequence runner that accepts ordered Unity frames
- Ground-truth evaluation for tracked trajectories where labels are available
- Cleaner handoff fields for the PID/gimbal layer
```

### API / Unity Integration

Expose the pipeline to Unity through a lightweight API or file/socket bridge.

Planned file:

```text
src/api.py
```

The API should accept a Unity camera frame and return:

```json
{
  "target_found": true,
  "target_id": "Terminal_B",
  "x_px": 486,
  "y_px": 192,
  "predicted_x_px": 493,
  "predicted_y_px": 190,
  "confidence": 0.96
}
```

### Closed-Loop Demo

Once the PID and Unity teams connect their parts, test the full behaviour:

```text
Search -> detect -> identify -> align -> track -> disturb -> lose lock -> reacquire
```

## Current Limitations

The dataset and model are synthetic. Even if metrics look strong, the classifier must be tested on Unity-generated frames under different lighting, motion, camera exposure, blur, target scale, and noise.

The current CNN can confuse true and false beacon-like glows because they are visually similar in 32 x 32 crops. This is expected at this stage and should improve with temporal verification, more realistic data, and classifier tuning.

The Phase 7 tracker needs ordered frames from the same camera sequence. Random individual dataset images are useful for detector/classifier tests, but real tracking behaviour should be tested with video-like Unity frame streams.

The project still does not communicate with Unity, issue PID commands, rotate a gimbal, or close the control loop. No real tracking accuracy is claimed yet without ground-truth trajectory data.
