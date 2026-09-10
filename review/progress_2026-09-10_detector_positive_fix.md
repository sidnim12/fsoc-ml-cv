# Progress - 2026-09-10 Detector-Positive Fix

## Problem

Full-frame accepted recall was low even though candidate recall was high. The issue was a training/runtime mismatch: positive CNN patches were label-centered, but live inference uses detector-centered candidate crops.

## Fix

`src/prepare_unity_patches.py` now writes matched detector candidates as additional `correct` patches.

## Outputs

```text
Patch output: data/processed/official_1600x900_detector_patches
Total patches: 17403
Correct patches: 16264
False patches: 1139
Training output: outputs/training/official_1600x900_detector_positive
Checkpoint: models/checkpoints/official_1600x900_detector_positive/best_classifier.pt
Evaluation: outputs/unity-evaluation/official_1600x900_detector_positive_conf005
Report: outputs/reports/official_1600x900_detector_positive_report.md
Config: configs/unity_detector_positive.yaml
```

## Final Metrics

```text
Candidate recall: 0.994643
Accepted detection recall: 0.994643
Filtered MAE: 2.950367 px
Locked frames: 99.000000%
Effective FPS: 8.721445
```
