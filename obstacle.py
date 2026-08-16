import cv2
import time
import threading
import subprocess
import collections
from ultralytics import YOLO
import speech_recognition as sr

# ---------- MAIN SETUP ----------
model = YOLO("yolov8n_ncnn_model")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

RECORDINGS_DIR = "/home/raspy/recordings"
CLOSE_THRESHOLD = 0.35
CONF_THRESHOLD = 0.45
COOLDOWN_SECONDS = 4
REMINDER_SECONDS = 10
CONFIRM_FRAMES = 2
RECORD_HOLD = 5
RECORD_FPS = 6

last_warning = {}
last_close_object = None
last_announce_time = 0
last_spoken_class = None
last_spoken_direction = None
pending_class = None
pending_count = 0

# ---------- SPEECH QUEUE ----------
# A single background worker speaks one message at a time, so the main
# detection loop and the voice-query listener can never talk over each other.
speech_cv = threading.Condition()
speech_items = collections.deque()  # each item: (text, is_hazard)


def speak(text, hazard=False):
    with speech_cv:
        if hazard:
            # A newer hazard message supersedes any hazard message still
            # waiting to be spoken, so speech doesn't lag behind stale state.
            # Query replies are left alone and never dropped.
            speech_items_list = [item for item in speech_items if not item[1]]
            speech_items.clear()
            speech_items.extend(speech_items_list)
        speech_items.append((text, hazard))
        speech_cv.notify()


def _speech_worker():
    while True:
        with speech_cv:
            while not speech_items:
                speech_cv.wait()
            text, _ = speech_items.popleft()
        subprocess.run(
            ["espeak-ng", "-s", "130", "-v", "en-us+f3", text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )


threading.Thread(target=_speech_worker, daemon=True).start()


def get_direction(x1, x2, frame_w):
    center = (x1 + x2) / 2
    if center < frame_w * 0.33:
        return "on your left"
    if center > frame_w * 0.66:
        return "on your right"
    return "ahead"


def new_writer():
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = f"{RECORDINGS_DIR}/rec_{ts}.mp4"
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), RECORD_FPS, (640, 480)), path


recognizer = sr.Recognizer()
mic = sr.Microphone()


def listen_for_query():
    global last_close_object
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
    print("Mic ready, listening for questions...")
    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
            print("Heard something, recognizing...")
            text = recognizer.recognize_google(audio).lower()
            print(f"You said: {text}")
            if "what kind of object" in text or "what is it" in text:
                if last_close_object:
                    speak(f"It looks like a {last_close_object}")
                else:
                    speak("Nothing detected right now")
        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            print("Heard audio but couldn't understand it")
        except sr.RequestError as e:
            print(f"Speech recognition error (check internet): {e}")


threading.Thread(target=listen_for_query, daemon=True).start()

# ---------- MAIN LOOP ----------
out = None
recording = False
last_detection_time = 0

print("Starting obstacle detection... press Ctrl+C in terminal to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame, verbose=False, conf=CONF_THRESHOLD)
    annotated = results[0].plot()

    h, w = frame.shape[:2]
    detected_this_frame = False

    largest_ratio_this_frame = 0
    largest_object_this_frame = None
    largest_box_this_frame = None

    for box in results[0].boxes:
        cls = model.names[int(box.cls)]
        x1, y1, x2, y2 = box.xyxy[0]
        width_ratio = (x2 - x1) / w
        detected_this_frame = True

        if width_ratio > largest_ratio_this_frame:
            largest_ratio_this_frame = width_ratio
            largest_object_this_frame = cls
            largest_box_this_frame = (x1, x2)

    if largest_ratio_this_frame > CLOSE_THRESHOLD:
        # Require the same class to persist for a couple of frames before
        # acting on it, so a single misclassified frame doesn't trigger an
        # announcement or reset the "still there" reminder timer.
        if largest_object_this_frame == pending_class:
            pending_count += 1
        else:
            pending_class = largest_object_this_frame
            pending_count = 1

        if pending_count >= CONFIRM_FRAMES:
            last_close_object = pending_class
            x1, x2 = largest_box_this_frame
            direction = get_direction(x1, x2, w)

            now = time.time()
            changed = (pending_class != last_spoken_class) or (direction != last_spoken_direction)
            past_cooldown = now - last_announce_time > COOLDOWN_SECONDS
            past_reminder = now - last_announce_time > REMINDER_SECONDS

            if past_cooldown and (changed or past_reminder):
                speak(f"{pending_class} {direction}", hazard=True)
                last_announce_time = now
                last_spoken_class = pending_class
                last_spoken_direction = direction
    else:
        pending_class = None
        pending_count = 0

    if detected_this_frame:
        last_detection_time = time.time()
        if not recording:
            out, path = new_writer()
            recording = True
            print(f"Recording started: {path}")

    if recording:
        out.write(annotated)
        if time.time() - last_detection_time > RECORD_HOLD:
            out.release()
            recording = False
            print("Recording stopped (no detections).")

    time.sleep(0.01)
