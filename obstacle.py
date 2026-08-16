import cv2
import time
import threading
import subprocess
from ultralytics import YOLO
import speech_recognition as sr

# ---------- MAIN SETUP ----------
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

RECORDINGS_DIR = "/home/raspy/recordings"
CLOSE_THRESHOLD = 0.35
COOLDOWN_SECONDS = 4
RECORD_HOLD = 5
RECORD_FPS = 6

last_warning = {}
last_close_object = None
last_announce_time = 0

def speak(text):
    threading.Thread(target=lambda: subprocess.run(
        ["espeak-ng", "-s", "130", "-v", "en-us+f3", text],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )).start()

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

    results = model(frame, verbose=False, conf=0.25)
    annotated = results[0].plot()

    h, w = frame.shape[:2]
    detected_this_frame = False

    largest_ratio_this_frame = 0
    largest_object_this_frame = None

    for box in results[0].boxes:
        cls = model.names[int(box.cls)]
        x1, y1, x2, y2 = box.xyxy[0]
        width_ratio = (x2 - x1) / w
        detected_this_frame = True

        if width_ratio > largest_ratio_this_frame:
            largest_ratio_this_frame = width_ratio
            largest_object_this_frame = cls

    if largest_ratio_this_frame > CLOSE_THRESHOLD:
        now = time.time()
        last_close_object = largest_object_this_frame
        if now - last_announce_time > COOLDOWN_SECONDS:
            speak("Object detected")
            last_announce_time = now
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
