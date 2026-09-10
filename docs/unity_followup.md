# Unity Follow-Up Notes

These notes summarize what Unity should improve before final 6000-9000 frame training.

## Dataset Target

Use 20-30 sequences at 1600x900. Each sequence should have 300 continuous PNG frames and one `labels.csv`.

A 20-sequence final set gives 6000 frames. A 30-sequence final set gives 9000 frames.

## Keep These 12 Scenario Types

```text
smooth_horizontal
smooth_vertical
diagonal_motion
curved_motion
speed_variation
disturbance_shake
beacon_dropout
reacquisition
dim_beacon
multiple_beacon
target_absent
false_beacon_star_heavy
```

Add more sequences by varying start position, speed, brightness, star density, dropout timing, shake strength, false beacon count, and background orientation.

## Important Fixes

### Speed Variation

Current `sequence_005` has non-smooth target motion:

```text
mean visible step: 132.33 px/frame
max visible step: 377.41 px/frame
large steps above 120 px/frame: 110
```

This makes stable lock unfairly hard. Speed variation should be fast but continuous, not teleport-like.

### Reacquisition

Make the sequence clearly demonstrate:

```text
track -> dropout/lost -> local search -> beacon visible again -> reacquire -> track
```

Labels must remain aligned with the exported frame IDs.

### Ground Truth Use

Unity may use `WorldToViewportPoint` to create labels and calculate evaluation truth, but not for live control. Live control should use the FastAPI output only.

## Label Format

Required columns:

```csv
sequence_id,frame_id,timestamp_s,image_path,scenario,target_present,target_id,target_x,target_y
```

Recommended extra columns:

```csv
width,height,beacon_radius,beacon_on,occluded,faded,false_candidate_count,camera_pan,camera_tilt
```

Coordinates should use top-left image origin:

```text
x increases right
y increases down
```

## Official 9k Dataset Inspection Result

The final dataset passed basic ML/CV inspection:

```text
Sequences: 30
Frames: 9000
Resolution: 1600x900
Frame continuity: pass
Label/image count: pass
Smoothness: pass overall
```

Note: `sequence_020` contains one labelled motion step above 120 px. It is not a blocker, but it should be reviewed visually if final lock behavior looks strange in that sequence.
