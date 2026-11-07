# Obstacle Detector: Fix Voice Overlap, Repetition, and Detection Accuracy

## Context

`obstacle.py` is a Raspberry Pi assistive-navigation script: webcam + YOLOv8n
detect nearby obstacles, `espeak-ng` speaks alerts, and a background thread answers
spoken "what is it?" queries. Three problems have been observed in real use:

1. Voice lines overlap each other.
2. Object identification isn't accurate.
3. Voice lines repeat too much.

Root causes found by reading the code:

- **Overlap**: `speak()` spawns a brand-new thread + `espeak-ng` subprocess on every
  call, with no lock or queue. The main detection loop and the "what is it?" listener
  thread can both trigger speech at the same time.
- **Repetition**: while any object stays within `CLOSE_THRESHOLD`, the loop repeats a
  generic `"Object detected"` every `COOLDOWN_SECONDS` (4s), regardless of whether
  it's the same object still sitting there. `get_direction()` is defined but never
  called, so announcements carry no object name or direction.
- **Inaccuracy**: the script loads `yolov8n.pt` directly, even though a pre-exported
  `yolov8n_ncnn_model/` (same 80 COCO classes, confirmed via its `metadata.yaml`)
  sits right next to it unused. Confidence threshold is a permissive `conf=0.25`,
  and there's no debounce — a single-frame misclassification is acted on immediately.

Scope for this pass is fixing these three issues plus wiring up spoken object+direction.
Two ideas raised during discussion — live scene description via a vision-language
model, and detecting non-COCO obstacle classes (curbs, poles, stairs) via custom
training — are explicitly **deferred** to a future spec; this pass only uses what
YOLOv8n's existing COCO classes already cover.

## Steps

### 0. Branch + planning notes (before any code changes)

- Create and switch to branch `fix/voice-overlap-and-accuracy`.
- Write this design to `docs/2026-08-16-voice-and-accuracy-fixes.md` and commit
  it before starting the code changes below.

### 1. Speech: single-consumer queue instead of fire-and-forget threads

Replace `speak()`'s current "new thread + subprocess per call" with:
- A `queue.Queue` for pending speech text.
- One dedicated daemon worker thread started at startup that pulls from the
  queue and runs `espeak-ng` synchronously (blocking), one message at a time —
  guaranteeing no two `espeak-ng` processes ever run concurrently.
- `speak()` becomes just `queue.put(...)` and returns immediately, so existing
  call sites (main loop, `listen_for_query` thread) don't need to change how
  they call it.
- Hazard-type messages (obstacle announcements) are tagged; when a new hazard
  message is enqueued, any not-yet-spoken hazard message still sitting in the
  queue is dropped (superseded) so speech doesn't lag behind stale state.
  "What is it?" query replies are never dropped.

### 2. Announcement logic: speak on change, not on a blind timer

Replace "re-announce every `COOLDOWN_SECONDS` while anything is close" with:
- Track `last_spoken_class` and `last_spoken_direction`.
- Announce when the closest object's class or direction changes.
- If it's the same object/direction as last announced, only repeat as a
  "still there" reminder after a longer `REMINDER_SECONDS` (e.g. 10s), not
  every 4s.
- `COOLDOWN_SECONDS` remains as a minimum floor between any two announcements,
  to absorb frame-to-frame flicker.

### 3. Accuracy: NCNN model, stricter threshold, debounce flicker

- Change `model = YOLO("yolov8n.pt")` to `model = YOLO("yolov8n_ncnn_model")`
  — same 80 COCO classes, faster CPU inference on the Pi, less motion blur
  per frame as a side effect of higher achievable frame rate.
- Raise `conf` from `0.25` to a stricter default (e.g. `0.45`).
- Require a candidate "closest object" class to persist for 2 consecutive
  frames before it's accepted as the current close object — filters one-off
  misclassification noise and reduces flapping in step 2's change-detection.

### 4. Speech content: object + direction

Wire the existing (currently unused) `get_direction(x1, x2, frame_w)` into the
announcement path: speak `f"{object_class} {direction}"`, e.g. `"person ahead"`,
`"chair on your left"` — replacing the generic `"Object detected"`.

## Verification

This runs on Raspberry Pi hardware (camera, mic, `espeak-ng`) not available in
the dev environment, so verification is manual, on-device, after syncing the
branch to the Pi:

- Single object slowly approaching from center: confirm one announcement per
  meaningful change, not one every 4 seconds.
- Two different objects appearing in sequence: confirm both are announced with
  correct left/right/ahead direction.
- One object held in place for 15+ seconds: confirm only an occasional
  "still there" reminder, not constant repetition.
- Trigger a "what is it?" voice query while a hazard announcement is queued:
  confirm both play fully, in order, without overlapping audio.
- Spot-check detections against `conf=0.45` + NCNN model for fewer obviously
  wrong labels compared to the current `yolov8n.pt` + `conf=0.25` baseline.

## Deferred (future spec)

- Live scene description via a vision-language model.
- Detecting non-COCO obstacle classes (curbs, poles, stairs, potholes) via
  custom data collection + retraining.
