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

## Work Completed On September 7, 2026

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
    training/
  src/
    generate_dataset.py
    preprocessing.py
    candidate_detector.py
    prepare_patches.py
    beacon_classifier.py
    train_classifier.py
    tracker.py
    pipeline.py
    evaluate.py
    api.py
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

Explores patch dataset balance, split counts, patch previews, training history, test metrics, and training artifacts.

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

## Future Plan

### Next Phase: Improve CNN Performance

The first CNN is working, but the test recall is not perfect. Next improvements should focus on:

```text
- Better false-patch diversity
- More realistic Unity-style lighting variation
- More target-absent hard negatives
- More balanced train/validation/test examples
- Experimenting with crop size and augmentation strength
- Exporting the final classifier to ONNX
```

### Temporal Beacon Verification

After the classifier, add a temporal verification module that checks whether the selected candidate follows the expected beacon blinking pattern. This helps reject stars and static false lights.

Planned file:

```text
src/pipeline.py or src/temporal_verifier.py
```

### Kalman Tracking

Implement image-space smoothing and prediction for the beacon centre.

Planned file:

```text
src/tracker.py
```

Expected output:

```text
current x, current y, predicted x, predicted y, lock state
```

### Full ML/CV Pipeline

Combine preprocessing, candidate detection, CNN classification, temporal verification, and Kalman prediction into one callable pipeline.

Planned file:

```text
src/pipeline.py
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
