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

### Phase 8: Unity Dataset Adaptation And Baseline Evaluation

Implemented in:

```text
src/unity_dataset_adapter.py
src/evaluate_unity_sequence.py
src/prepare_unity_patches.py
configs/unity.yaml
tests/test_unity_dataset_adapter.py
tests/test_unity_evaluation.py
```

This phase adapts a continuous Unity image sequence to the ML/CV evaluation format without copying, moving, resizing, or modifying the original Unity frames.

Unity sequence discovered locally:

```text
data/raw/unity/smooth_horizontal01/sequence_001/
```

Expected Unity folder shape:

```text
data/raw/unity/
  <scenario-folder>/
    <sequence-folder>/
      frame_000000.png
      frame_000001.png
      ...
      labels.csv
```

Supported image formats:

```text
.png
.jpg
.jpeg
```

Supported label files:

```text
labels.csv
labels.xlsx
metadata.csv
metadata.xlsx
```

Preferred label fields:

```text
sequence_id, frame_index, timestamp_s, image_path, scenario,
width, height, target_visible, target_id, target_x, target_y
```

Accepted aliases include:

```text
frame_id -> frame_index
filename -> image_path
target_present -> target_visible
cx_px -> target_x
cy_px -> target_y
```

Coordinate origin must be explicit:

```powershell
--coordinate-origin top-left
--coordinate-origin bottom-left
```

For bottom-left Unity coordinates, the evaluator converts to OpenCV top-left image coordinates:

```text
opencv_x = unity_x
opencv_y = image_height - unity_y
```

Baseline evaluation command used:

```powershell
.\.venv\Scripts\python.exe src\evaluate_unity_sequence.py --sequence-dir data\raw\unity\smooth_horizontal01\sequence_001 --config configs\default.yaml --checkpoint models\checkpoints\best_classifier.pt --fps 30 --coordinate-origin top-left --output outputs\unity-evaluation\smooth_horizontal_01 --expected-count 300 --expected-width 640 --expected-height 480 --device cpu
```

Unity-ready batch command:

```powershell
.\.venv\Scripts\python.exe src\evaluate_unity_sequence.py --batch --sequence-dir data\raw\unity --config configs\unity.yaml --fps 30 --coordinate-origin top-left --output outputs\unity-evaluation --output-scale 2 --device cpu
```

This command discovers every sequence under `data/raw/unity`, auto-detects frame resolution, validates labels, evaluates each sequence, and writes combined metrics:

```text
outputs/unity-evaluation/final_metrics.csv
outputs/unity-evaluation/final_summary.json
```

`--output-scale 2` only enlarges the annotated MP4 for easier presentation viewing. It does not resize, overwrite, or relabel the source Unity frames. True higher-quality evaluation still requires Unity to export higher-resolution frames and matching coordinates.

Dataset validation result:

```text
Images: 300 PNG
Resolution: 640 x 480
Channels: 3
Frame range: 0 -> 299
Continuous frames: true
Duplicate frame numbers: 0
Duplicate image files: 0
Unreadable images: 0
Labels found: labels.csv
Ground truth available: true
Unity Editor UI suspected: false
```

Baseline detection result on the first Unity sequence:

```text
Visible target frames: 300
Candidate recall: 0.096667
Accepted-detection recall: 0.0
Missed detections: 300
Target-found rate: 0.563333
```

Baseline tracking result:

```text
Time to first lock: 2 frames / 0.066667 s
Locked frames: 84.333333%
Coasting frames: 10.666667%
Lost frames: 0.333333%
Maximum consecutive missed frames: 9
Filtered MAE: 288.899709 px
```

Domain-gap finding:

```text
Average candidates per frame: 8.086667
Frames with multiple candidates: 300
Average correct-beacon probability across candidates: 0.112807
Selected average correct-beacon probability: 0.581765
Most common failure: selected_candidate_far_from_ground_truth
```

The current Python-trained CNN frequently locks onto bright Unity artifacts instead of the labelled beacon. This is useful baseline evidence and should be fixed with detector tuning, better Unity-labelled data, and later fine-tuning. This phase intentionally does not retrain the CNN.

Annotated video legend:

```text
Red circle/trail: Unity ground-truth beacon
Yellow circle/trail: selected ML/CV candidate
Blue trail / F marker: Kalman-filtered track
Green M marker: measured selected candidate
Magenta P marker: tracker prediction
Gray rings: other bright candidates detected by OpenCV
```

If the yellow selected trail stays near the Earth edge while the red ground-truth trail moves elsewhere, that is not a drawing bug. It shows the current Unity-domain failure: bright scene artifacts are being selected instead of the true beacon.

Phase 8 outputs:

```text
data/processed/unity/smooth_horizontal_01/manifest.csv
outputs/unity-evaluation/smooth_horizontal_01/dataset_validation.json
outputs/unity-evaluation/smooth_horizontal_01/normalized_manifest.csv
outputs/unity-evaluation/smooth_horizontal_01/frame_predictions.csv
outputs/unity-evaluation/smooth_horizontal_01/detection_metrics.json
outputs/unity-evaluation/smooth_horizontal_01/tracking_metrics.json
outputs/unity-evaluation/smooth_horizontal_01/performance_metrics.json
outputs/unity-evaluation/smooth_horizontal_01/domain_gap_report.json
outputs/unity-evaluation/smooth_horizontal_01/evaluation_summary.json
outputs/unity-evaluation/smooth_horizontal_01/annotated_tracking.mp4
outputs/unity-evaluation/smooth_horizontal_01/trajectory_comparison.png
outputs/unity-evaluation/smooth_horizontal_01/confidence_over_time.png
outputs/unity-evaluation/smooth_horizontal_01/coordinate_error_over_time.png
outputs/unity-evaluation/smooth_horizontal_01/lock_state_timeline.png
outputs/unity-evaluation/smooth_horizontal_01/failure_montage.png
outputs/unity-evaluation/final_metrics.csv
outputs/unity-evaluation/final_summary.json
```

### Unity 2400-Frame Training Pass

New Unity dataset checked locally:

```text
data/raw/unity/cont_dataset_2400/unity_base_2400/
```

Contents:

```text
8 sequences
300 PNG frames per sequence
2400 total frames
1280 x 720 resolution
8 labels.csv files
```

Scenarios:

```text
smooth_horizontal
smooth_vertical
diagonal
curved
speed_variation
disturbance
beacon_dropout
reacquisition
```

Label timing finding:

```text
The exported label at frame t best matches image frame t+1.
Evaluation/training for this dataset uses --label-frame-offset 1.
```

Unity-domain patch preparation:

```powershell
.\.venv\Scripts\python.exe src\prepare_unity_patches.py --root data\raw\unity\cont_dataset_2400\unity_base_2400 --output data\processed\unity_base_2400_patches --config configs\unity.yaml --label-frame-offset 1 --crop-size 64 --match-tolerance 24 --snap-radius 64 --snap-min-intensity 180 --max-negatives-per-frame 5 --max-negative-ratio 3 --seed 42
```

Patch output:

```text
data/processed/unity_base_2400_patches/patch_labels.csv
data/processed/unity_base_2400_patches/dataset_summary.json
```

Patch counts:

```text
Total patches: 5624
Correct patches: 1406
False patches: 4218
Train: 1020 correct, 3060 false
Validation: 184 correct, 552 false
Test: 202 correct, 606 false
```

Unity-domain classifier training:

```powershell
.\.venv\Scripts\python.exe src\train_classifier.py --metadata data\processed\unity_base_2400_patches\patch_labels.csv --data-root data\processed\unity_base_2400_patches --output-dir outputs\training\unity_base_2400 --checkpoint-dir models\checkpoints\unity_base_2400 --epochs 15 --batch-size 64 --learning-rate 0.001 --patience 4 --device cpu
```

Training outputs:

```text
models/checkpoints/unity_base_2400/best_classifier.pt
models/checkpoints/unity_base_2400/last_classifier.pt
outputs/training/unity_base_2400/history.csv
outputs/training/unity_base_2400/test_metrics.json
outputs/training/unity_base_2400/classification_report.json
outputs/training/unity_base_2400/confusion_matrix.png
outputs/training/unity_base_2400/training_curves.png
outputs/training/unity_base_2400/sample_predictions.png
```

Patch-level test result:

```text
Accuracy: 1.0000
Precision: 1.0000
Recall: 1.0000
F1-score: 1.0000
ROC-AUC: 1.0000
```

Full-frame evaluation with the Unity-trained checkpoint:

```powershell
.\.venv\Scripts\python.exe src\evaluate_unity_sequence.py --batch --sequence-dir data\raw\unity\cont_dataset_2400\unity_base_2400 --config configs\unity.yaml --checkpoint models\checkpoints\unity_base_2400\best_classifier.pt --fps 30 --coordinate-origin top-left --output outputs\unity-evaluation\unity_base_2400_trained --output-scale 1 --match-tolerance 24 --label-frame-offset 1 --device cpu
```

Before/after full-frame result:

```text
Before Unity training:
  Candidate recall: 0.033074
  Accepted recall: 0.000000
  Filtered MAE: 475.900149 px

After Unity training with corrected label offset:
  Candidate recall: 0.392973
  Accepted recall: 0.144187
  Filtered MAE: 59.11329 px
```

Interpretation:

```text
The Unity-trained classifier improves full-frame tracking a lot, but the remaining bottleneck is candidate detection.
If OpenCV does not produce a candidate near the true beacon, the CNN cannot select it.
Next work should tune Unity candidate detection and confirm/fix Unity label export timing.
```

If labels are missing, the same evaluator still runs inference and diagnostics, marks quantitative ground-truth metrics as `null`, and creates:

```text
outputs/unity-evaluation/<sequence>/required_labels_template.csv
```

Why this one sequence is not used for training:

```text
One continuous sequence has strong frame-to-frame similarity.
Splitting it into train/test frames would leak temporal information.
It is better used as a held-out baseline evaluation sequence.
```

Phase 8 tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_unity_dataset_adapter.py tests\test_unity_evaluation.py -v
.\.venv\Scripts\python.exe -m pytest -v
```

Result:

```text
25 targeted Phase 8/Unity tests passed
58 total project tests passed
```

## Current Project Structure

```text
fsoc-ml-cv/
  configs/
    default.yaml
    unity.yaml
  data/
    labels/
      labels.csv
    raw/
      python-generated/
      unity/
    processed/
      unity/
        smooth_horizontal_01/
      unity_base_2400_patches/
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
    unity-evaluation/
      smooth_horizontal_01/
      unity_base_2400/
      unity_base_2400_trained/
    tracking-test/
    training/
  review/
    progress_2026-09-07.md
    progress_2026-09-08.md
    progress_2026-09-09.md
  src/
    generate_dataset.py
    preprocessing.py
    candidate_detector.py
    prepare_patches.py
    prepare_unity_patches.py
    beacon_classifier.py
    train_classifier.py
    temporal_verifier.py
    tracker.py
    unity_dataset_adapter.py
    evaluate_unity_sequence.py
    pipeline.py
    evaluate.py
    api.py
  tests/
    test_detector.py
    test_pipeline.py
    test_temporal_verifier.py
    test_tracker.py
    test_unity_dataset_adapter.py
    test_unity_evaluation.py
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

Explores patch dataset balance, split counts, patch previews, training history, test metrics, training artifacts, the Phase 6 single-frame inference output, the Phase 7 tracking smoke summary, and the Phase 8 Unity baseline evaluation.

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

Unity baseline evaluation:

```text
data/processed/unity/smooth_horizontal_01/manifest.csv
outputs/unity-evaluation/smooth_horizontal_01/
```

## Future Plan

### Next Phase: Unity Domain Adaptation

Use the Phase 8 baseline results to improve Unity-frame performance without contaminating evaluation data.

Expected additions:

```text
- Review Unity render settings and remove bright artifacts where possible
- Tune preprocessing/detector thresholds for Unity frames
- Ask Unity to export more varied labelled sequences
- Fine-tune only after collecting multiple independent sequences
- Keep at least one labelled sequence held out for honest evaluation
```

### API / Unity Integration

Expose the pipeline to Unity through a lightweight API or file/socket bridge.

Planned file:

```text
src/api.py
```

### PID / Gimbal Integration

Connect filtered and predicted pixel coordinates to the PID controller for pan/tilt movement.

Expected handoff fields:

```text
target_found
filtered_x_px
filtered_y_px
predicted_x_px
predicted_y_px
control_error_x
control_error_y
lock_state
```

The future API should accept a Unity camera frame and return:

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

The CNN was trained on Python-generated synthetic images and has now been tested on one labelled Unity sequence. The Phase 8 baseline shows a clear domain gap, so the classifier must be improved using more varied Unity-generated frames before it is trusted.

The current CNN can confuse true and false beacon-like glows because they are visually similar in 32 x 32 crops. This is expected at this stage and should improve with temporal verification, more realistic data, and classifier tuning.

The Phase 7 tracker needs ordered frames from the same camera sequence. Phase 8 provides an offline sequence evaluator, but live Unity communication is still not implemented.

The project still does not communicate with Unity, issue PID commands, rotate a gimbal, or close the control loop. Current Unity metrics are baseline offline metrics from one sequence, not final system performance.

### Official Unity 1600x900 V2 Evaluation

The official Unity 1600x900 dataset was validated with 12 sequences and 3600 total frames. Each sequence contains 300 continuous PNG frames and a matching labels.csv file.

Scenarios covered: smooth_horizontal, smooth_vertical, diagonal_motion, curved_motion, speed_variation, disturbance_shake, beacon_dropout, reacquisition, dim_beacon, multiple_beacon, target_absent, and false_beacon_star_heavy.

Baseline evaluation completed on all 3600 frames:

```text
Average candidate recall: 0.912883
Average accepted recall: 0.377689
Average filtered MAE: 303.05556 px
```

Unity patch preparation produced:

```text
Total patches: 10772
Correct patches: 2693
False patches: 8079
```

The split logic was updated so target-absent-only sequences do not become the validation set. CNN training completed with early stopping after 12 epochs:

```text
Test accuracy: 0.8639
Test precision: 0.7034
Test recall: 0.7876
Test F1-score: 0.7432
Test ROC-AUC: 0.9299
```

Full-frame trained evaluation completed on all 3600 frames:

```text
Average candidate recall: 0.999650
Average accepted recall: 0.074614
Average filtered MAE: 5.919378 px
Average locked-frame percentage: 44.583333
Average effective processing FPS: 8.093595
```

Interpretation: OpenCV candidate recall is now excellent and filtered localization error is very low. The remaining weakness is accepted detection recall/confidence gating, especially reacquisition and speed-variation behavior. Next tuning should focus on confidence thresholds, temporal verification, and tracker lock-state parameters rather than basic candidate detection.

### Unity 1600x900 Tuning Result

After CNN training, a confidence and tracking sweep was run on representative sequences. Very loose settings increased lock but produced large errors on speed variation, so the balanced Unity settings are:

```yaml
classifier:
  confidence_threshold: 0.25
temporal:
  min_confirmations: 2
tracker:
  association_gate_px: 120
```

Full 3600-frame tuned evaluation:

```text
Average candidate recall: 0.999650
Average accepted recall: 0.245962
Average filtered MAE: 6.288764 px
Average locked-frame percentage: 50.805556
Average effective processing FPS: 7.981692
```

Compared with the previous trained evaluation, accepted recall improved from 0.074614 to 0.245962 and locked frames improved from 44.583333% to 50.805556%, while filtered error stayed low.

Remaining issue: sequence_005 speed variation still has 0% LOCKED despite low pixel error, and sequence_008 reacquisition still has very low accepted recall. The next improvement should tune state-transition/lock criteria instead of lowering confidence too aggressively.

### Speed-Variation Motion Diagnostic

The speed-variation sequence has non-smooth ground-truth motion. Its labels show large frame-to-frame target jumps, so forcing the tracker to lock more aggressively increases false/unstable tracking error.

Current diagnostic for sequence_005:

```text
Mean visible target step: 132.33 px/frame
Maximum visible target step: 377.41 px/frame
Large motion steps above 120 px/frame: 110
```

Because of this, the safer behavior is to keep accurate ACQUIRING measurements instead of forcing LOCKED state on discontinuous motion. Future Unity exports should make speed variation fast but physically continuous if the demo expects stable lock.

### Unity Handoff And Review Reports

Backend API implementation is owned by the backend developer, but the ML/CV repo now documents the exact contract they should connect to:

```text
docs/api_contract.md
docs/unity_followup.md
src/summarize_unity_results.py
```

Generate a compact final Unity metrics report with:

```bash
python src/summarize_unity_results.py --evaluation-root outputs/unity-evaluation/official_1600x900_v2_tuned_conf025 --output outputs/reports/official_1600x900_v2_tuned_report.md --title "Official Unity 1600x900 V2 Tuned Report"
```

Current report output:

```text
outputs/reports/official_1600x900_v2_tuned_report.md
```

The report highlights overall candidate recall, accepted detection recall, filtered pixel error, lock percentage, FPS, and sequence-level weak cases. Unity follow-up work should focus on smoother speed-variation labels, clearer reacquisition behavior, and final 6000-9000 frame export variations.

### Official 1600x900 9k Dataset Result

Final Unity dataset inspected under:

```text
data/raw/unity/official_1600x900/official_1600x900
```

Dataset validation:

```text
Sequences: 30
Frames: 9000
Resolution: 1600x900
Frames per sequence: 300
Continuity: all sequences continuous
Smoothness: mostly smooth; sequence_020 has one large labelled step above 120 px
```

Patch preparation:

```text
Total patches: 9271
Correct patches: 8132
False patches: 1139
Train split: 6509 correct, 911 false
Validation split: 814 correct, 114 false
Test split: 809 correct, 114 false
```

Patch-level CNN training output:

```text
Checkpoint: models/checkpoints/official_1600x900_final/best_classifier.pt
Training output: outputs/training/official_1600x900_final
Test accuracy: 1.0000
Test precision: 1.0000
Test recall: 1.0000
Test F1-score: 1.0000
ROC-AUC: 1.0000
```

Full-frame Unity evaluation output:

```text
Evaluation output: outputs/unity-evaluation/official_1600x900_final_trained
Report: outputs/reports/official_1600x900_final_report.md
Average candidate recall: 0.994643
Average accepted detection recall: 0.078690
Average filtered MAE: 2.277340 px
Average locked-frame percentage: 39.022222
Average effective processing FPS: 8.189958
```

Interpretation: OpenCV candidate generation remains strong and localization error is very low when accepted. The main remaining weakness is full-frame acceptance/classifier confidence, because many correct candidates are still rejected as `below_confidence_threshold` despite strong patch-level test results.

### Detector-Centered Positive Patch Improvement

The low accepted recall was traced to a training/inference mismatch: the CNN was trained mostly on label-centered positive crops, while runtime inference classifies detector-centered candidate crops. `src/prepare_unity_patches.py` now adds matched detector candidates as additional `correct` training patches.

Updated patch/training/evaluation outputs:

```text
Patch output: data/processed/official_1600x900_detector_patches
Total patches: 17403
Correct patches: 16264
False patches: 1139
Training output: outputs/training/official_1600x900_detector_positive
Checkpoint: models/checkpoints/official_1600x900_detector_positive/best_classifier.pt
Evaluation output: outputs/unity-evaluation/official_1600x900_detector_positive_conf005
Report: outputs/reports/official_1600x900_detector_positive_report.md
```

Full-frame result after the fix:

```text
Average candidate recall: 0.994643
Average accepted detection recall: 0.994643
Average filtered MAE: 2.950367 px
Average locked-frame percentage: 99.000000
Average effective processing FPS: 8.721445
```

This is the current best ML/CV configuration for the 1600x900 Unity prototype. Use `configs/unity_detector_positive.yaml` with the detector-positive checkpoint for backend integration tests.
