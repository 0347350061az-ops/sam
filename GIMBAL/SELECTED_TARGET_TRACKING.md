# Track any selected target

From the project directory, run:

```powershell
.\.venv\Scripts\python.exe scripts\camera.py --target-mode auto --engine enhanced
```

Drag a tight box around the desired target in LIVE PREVIEW. Press R to select
a different target. Enhanced tracking is the default. Do not pass
`--target-mode face` when selecting arbitrary objects.

The tracker stores the selected target's mask and image structure. After loss,
it searches locally and then across the image, including scale and quarter-turn
rotation variants. A candidate must beat competing candidates and pass SAM
verification on three observations before tracking resumes. Search continues
after prolonged occlusion without requiring reselection.

This uses visual templates rather than learned identity recognition. Similar
objects, textureless targets, major viewpoint changes, or substantial lighting
changes can prevent reliable recovery. Generic SAM masks can also expand into
touching regions; generic mode does not provide the face detector's boundary
constraint.

Kalman filtering is now off by default for handheld targets. The supplied
bottle recording showed motion gating rejecting correct SAM masks while the
bottle tilted and reversed direction. Image structure, mask stability, area
checks, optical flow, and confirmed recovery remain active. K can enable the
optional motion filter. With it enabled, two consistent strong observations
can reset a stale prediction. Nested scale proposals at the same location
count as one candidate; distinct locations still compete.

Validation: `scripts/check_hybrid.py` (25 checks) and
`scripts/check_hybrid_gpu.py` (real SAM on a synthetic object scene with full
occlusion and a shifted return). `scripts/check_hybrid_scenarios.py` also checks
fast movement, partial/full occlusion, and a same-color distractor. Webcam
accuracy still needs live evaluation.

The supplied screen recording was cropped to LIVE PREVIEW and replayed at
640x480, sampling every third frame from recording frame 210. Before this
default change, 13/78 samples had a target mask. Afterward, all 67 samples
through frame 408 (bottle visible) remained confirmed; all 8 samples from frame
420 (bottle absent) were invalid. Sampled overlay frames were visually checked
for bottle alignment. These are replay/state checks, not annotated mask IoU
or a claim about general tracking accuracy. The remaining 3 samples cover the
exit transition. The clip does not contain a return after exit.

`scripts/check_recorded_target.py` reproduces this replay and exports a JSON
report and overlay video. Local output: `runtime/recording_review/verified/`.

Return confirmation now keeps a separate provisional SAM state: after a global
proposal passes ambiguity checks, SAM follows it for three verified observations
without requiring a fresh template peak every frame. Failed candidates discard
that state; only confirmed recovery replaces the accepted state. Initial search
also checks competing orientations before starting confirmation.

`scripts/check_recorded_return.py runtime/recording_review/live.avi` constructs
an exit/return fixture from the supplied recording: after the exit and a blank
interval, earlier visible bottle frames return translated 180 pixels. Before
the confirmation fix, zero return samples recovered; afterward the bottle
recovered in 1.3 seconds with the same target ID. This is a constructed test,
not a naturally recorded exit and return. Output:
`runtime/recording_review/return_verified/overlay.mp4`.

Recovery proposals now use chromatic Lab structure for targets with appreciable
color, retaining grayscale search for neutral targets. This prevents similar
grayscale wall patterns from crowding out the bottle's green cap/label. Color
guides discovery; SAM stability, grayscale structure verification, ambiguity
checks, and three observations are still required for recovery. Verified views
retain their color crops. `recovery_color: false` restores grayscale proposals.

Natural recording `Recording 2026-11-13 141059.mp4`: crop `(7,35,800,600)`,
resize to 640x480, initial box `(180,55,285,385)` on recording frame 0, stride 3.
Grayscale A/B replay did not recover after loss. Chromatic search recovered at
frame 270 (9 seconds), then tracked through the end. Sampled overlays were
visually checked for bottle alignment rather than accepting status alone.
The replay reselects at its first frame; it is not an exact reproduction of
the original mouse gesture or live dropped-frame timing.
Outputs: `runtime/review_141059/bottle_before/` and
`runtime/review_141059/bottle_after/`. Add `--grayscale-search` to
`scripts/check_recorded_target.py` to reproduce the A/B baseline.

If whole-object matching finds no candidates, recovery now searches 65% parts
of the target along its long axis, with denser scales and alternating 0/±15
degree tilts. The match is extrapolated into a full target box and clipped to
the image. SAM still verifies that box through the existing confirmation path;
a part match alone does not authorize tracking. The original reference remains
immutable while redundant trusted views refresh the recent crop.

The partial-return unit check covers a target with its lower part outside the
image. The natural 14:10:59 recording and GPU global-recovery smoke still pass.
At that stage the failure log contained no camera frames. The subsequent
`--record-raw` captures enabled the direct investigation described below.

## Fix verified on actual raw reentry recordings

The next recordings were available locally and showed the failure directly:
`runtime/reentry_20261113_142713.avi` and `runtime/reentry_20261113_142625.avi`,
each with capture timestamps. The failing live log showed search steps of
1.6–4.3 seconds, skipping most of the returning bottle's visible interval.

The default app now loads the official torchvision SSDlite320 MobileNet V3
COCO detector once, alongside SAM. At selection, a detection must overlap
the selected SAM region by at least 0.4 IoU before its category can guide
discovery. When the target is lost, this detector proposes regions across
the whole image. The category and detector score never confirm identity:
the existing SAM stability, appearance, spatial continuity and three-frame
confirmation gates remain required. Detector memory is separate from SAM.
Selections without a matching detector category retain generic recovery.
Generic template search uses 320 pixels and two views to reduce latency;
small targets retain source resolution so they do not disappear below the
minimum template size. The optional face mode remains a separate explicit choice.

Weights: `data/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth`, fetched
from the official PyTorch model URL. No runtime weight download is attempted.
`configs/hybrid.json` enables `object_discovery`; setting it false runs the
generic recovery alone. A missing enabled model gives a clear startup error.

`scripts/check_reentry_live.py` uses the actual capture timestamps and advances
to the newest available frame after each measured processing step. This models
the live worker's dropped frames rather than slowing the clip to inference speed.
With the default app configuration on the RTX 3050:

| Raw capture | Selected frame | Recovered frame | Mean discovery | p95 discovery |
| --- | ---: | ---: | ---: | ---: |
| 14:27:13, worker CUDA stream | 487 | 680 | 38 ms | 43 ms |
| 14:26:25 | 146 | 416 | 83 ms | 104 ms |

The same target ID survives. Initial replay on the default CUDA stream also
recovered at frame 685 with 75/95 ms mean/p95 discovery. The final replay uses
the worker's separate priority stream after synchronizing model initialization.
The latest recording also passes the explicit
no-target interval 660–670 and recovery interval 675–745. Saved recovery masks
were visually checked on the bottle. These measurements are replay results,
not a guarantee for every class, scene or viewpoint. The detector covers COCO
categories; arbitrary targets still depend on generic image matching.
Outputs: `runtime/latest_review/live_replay/` and
`runtime/latest_review/previous_live_replay/`.

## Identity guard when targets become distant

The 14:44 live log repeatedly accepted masks whose bounds changed from a tall
narrow target to nearly square regions, even with grayscale similarity around
0.76. Tracking and recovery now also check immutable foreground support:

- Shape elongation uses mask covariance eigenvalues, so uniform shrinking and
  in-plane rotation do not change it. Accepted elongation must stay within
  0.55–1.8 times the original selection's elongation.
- If the original foreground contains enough saturated color, smoothed hue
  histograms must intersect by at least 0.35. Neutral original targets have no
  hue requirement. Fewer than 32 foreground pixels are insufficient evidence.
- Both checks run before accepting a continuous mask or a recovery proposal,
  and before adding a view to the appearance gallery. Rejected candidates keep
  control on HOLD. The original shape/color reference is never updated.
- Verified partial visibility can still be displayed as unconfirmed without
  changing the reference or supplying a control center.

Thresholds are `identity_shape_min`, `identity_shape_max`, and
`identity_hue_min` in `configs/hybrid.json`. These are explicit image checks,
not learned identity recognition or calibrated confidence. Two objects with
indistinguishable appearance/shape can still be ambiguous.

Tests cover shrinking/rotating the selected target, a same-color round object,
a different-color elongated object, refusal to learn a rejected view, and a
high-stability/high-similarity wrong SAM mask in both tracking and recovery.
The raw 14:27:13 timing replay still reacquires the bottle, and synthetic scale,
partial visibility, and same-color distractor checks pass without a wrong lock.
No raw recording exists for the latest distant-target run; the exact new scene
has not been replayed or assigned an accuracy score.

## One automatic selection workflow

The default command requires no person/object choice. SAM interprets the drawn
region; discovery automatically associates a supported category with that region.
Unsupported categories retain generic tracking/recovery. Only an explicit
`--target-mode face` requests the separate face-only path.

For an automatically associated person, an original-shape mismatch may represent
a pose change. It is accepted only when the stored appearance still matches at
the gallery-update threshold, a fresh person detection overlaps the whole SAM
box by at least 0.6 IoU, and original color support passes. A wider elongation
range (0.25–4) then permits poses. A small face mask inside a larger person
detection cannot use this exception. Objects and unclassified selections retain
the stricter shape limits. Accepted pose views may enter the bounded gallery;
the original identity remains immutable. The user selects no profile manually.

Tests include person-pose acceptance, rejection without whole-person evidence,
rejection of mismatched appearance, and preserving the object guard. A raw
capture replay selecting the visible person at frame 660 automatically associated
category 1 and kept TRACKING for all 147 processed observations through frame 879.
The saved overlay was visually inspected. This checks that recording's visible
person, not all poses or identity discrimination between similar people. Output:
`runtime/auto_selection/person/`.
