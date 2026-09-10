# Progress - 2026-09-10 Official 9k Dataset

## Dataset Inspection

```text
Dataset root: data/raw/unity/official_1600x900/official_1600x900
Sequences: 30
Frames: 9000
Resolution: 1600x900
Continuity: all sequences continuous
Smoothness: mostly smooth; sequence_020 has one large step above 120 px
```

## Patch Preparation

```text
Output: data/processed/official_1600x900_patches
Total patches: 9271
Correct patches: 8132
False patches: 1139
Train: 6509 correct, 911 false
Validation: 814 correct, 114 false
Test: 809 correct, 114 false
```

## CNN Training

```text
Output: outputs/training/official_1600x900_final
Checkpoint: models/checkpoints/official_1600x900_final/best_classifier.pt
Test accuracy: 1.0000
Test precision: 1.0000
Test recall: 1.0000
Test F1-score: 1.0000
ROC-AUC: 1.0000
```

## Full-Frame Evaluation

```text
Output: outputs/unity-evaluation/official_1600x900_final_trained
Report: outputs/reports/official_1600x900_final_report.md
Successful sequences: 30/30
Average candidate recall: 0.994643
Average accepted detection recall: 0.078690
Average filtered MAE: 2.277340 px
Average locked-frame percentage: 39.022222
Average effective processing FPS: 8.189958
```

## Next ML/CV Fix

Patch-level training is excellent, but full-frame accepted recall is low. The next useful work is to tune candidate acceptance/classifier confidence behavior and inspect low-confidence accepted/rejected patches from the final dataset.
