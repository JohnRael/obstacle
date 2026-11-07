import os
import re
import cv2
import json
import time
import threading
import subprocess
import collections
import audioop
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo
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

# ---------- DATE / TIME / WEATHER SETTINGS ----------
TIMEZONE = "Asia/Manila"
# Weather location (decimal degrees). Approximate Alicia, Isabela - change
# these if the robot is used somewhere else.
LATITUDE = 16.78
LONGITUDE = 121.70
WEATHER_CACHE_SECONDS = 600  # reuse the last weather answer for 10 minutes

# ---------- SCENE IDENTIFICATION SETTINGS ----------
# How many seconds of recent detections are combined to guess the room.
SCENE_WINDOW_SECONDS = 4
# Minimum average evidence per frame before the object-based guess is trusted.
SCENE_MIN_SCORE = 1.0
# Optional Places365 scene classifier (see setup notes). If these two files
# are missing, the program just uses the object-based guess instead.
PLACES_DIR = "/home/raspy/models/places365"
PLACES_WEIGHTS = f"{PLACES_DIR}/resnet18_places365.pth.tar"
PLACES_LABELS = f"{PLACES_DIR}/categories_places365.txt"
PLACES_MIN_CONF = 0.35  # below this the answer is phrased as "not sure"

last_warning = {}
last_close_object = None
last_announce_time = 0
last_spoken_class = None
last_spoken_direction = None
pending_class = None
pending_count = 0

# Shared between the main detection loop and the voice-query thread.
latest_frame = None  # most recent raw camera frame
recent_frames = collections.deque()  # (timestamp, set of class names seen)
recent_lock = threading.Lock()

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


# ---------- DATE / TIME ANSWERS ----------
def _now():
    try:
        return datetime.now(ZoneInfo(TIMEZONE))
    except Exception:
        # tzdata missing - fall back to the Pi's own clock
        return datetime.now()


def answer_date():
    now = _now()
    return f"Today is {now.strftime('%A, %B')} {now.day}, {now.year}"


def answer_time():
    now = _now()
    hour12 = now.hour % 12 or 12
    # "A M" / "P M" with a space so espeak spells it out instead of
    # reading "AM" as the word "am".
    suffix = "A M" if now.hour < 12 else "P M"
    return f"It is {hour12}:{now.minute:02d} {suffix}"


# ---------- WEATHER ANSWER ----------
# Open-Meteo (https://open-meteo.com) is free and needs no API key.
# Codes are the standard WMO weather codes it returns.
WEATHER_WORDS = {
    0: "clear skies", 1: "mostly clear skies", 2: "partly cloudy skies",
    3: "overcast skies", 45: "fog", 48: "fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "heavy rain showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail",
}
_weather_cache = {"time": 0.0, "text": None}


def answer_weather():
    now = time.time()
    if _weather_cache["text"] and now - _weather_cache["time"] < WEATHER_CACHE_SECONDS:
        return _weather_cache["text"]

    params = urllib.parse.urlencode({
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": "temperature_2m,apparent_temperature,weather_code",
        "daily": "precipitation_probability_max",
        "timezone": "auto",
        "forecast_days": 1,
    })
    try:
        with urllib.request.urlopen(
                f"https://api.open-meteo.com/v1/forecast?{params}", timeout=5) as resp:
            data = json.load(resp)
        cur = data["current"]
        temp = round(cur["temperature_2m"])
        feels = round(cur["apparent_temperature"])
        sky = WEATHER_WORDS.get(cur["weather_code"], "unsettled weather")
        text = f"Right now it is {temp} degrees Celsius with {sky}"
        if abs(feels - temp) >= 2:
            text += f", and it feels like {feels}"
        rain = (data.get("daily", {}).get("precipitation_probability_max") or [None])[0]
        if rain is not None:
            text += f". The chance of rain today is {rain} percent"
    except Exception as e:
        print(f"Weather lookup failed: {e}")
        return "Sorry, I can't check the weather right now. Please check the internet connection"

    _weather_cache["time"] = now
    _weather_cache["text"] = text
    return text


# ---------- SCENE IDENTIFICATION ----------
# Method 1 (always available): guess the room from the objects YOLO has seen
# over the last few seconds. Each COCO class that hints at a room adds
# "evidence" for it. Only COCO's 80 classes are known, so it will say it's
# unsure when it doesn't see anything telling.
ROOM_CLUES = {
    "bed": {"bedroom": 3},
    "toilet": {"bathroom": 3},
    "toothbrush": {"bathroom": 2},
    "hair drier": {"bathroom": 2},
    "sink": {"kitchen": 1, "bathroom": 1},
    "refrigerator": {"kitchen": 3},
    "oven": {"kitchen": 2},
    "microwave": {"kitchen": 2},
    "toaster": {"kitchen": 2},
    "dining table": {"dining area": 2, "kitchen": 1},
    "bowl": {"kitchen": 1, "dining area": 1},
    "cup": {"kitchen": 0.5, "dining area": 0.5},
    "fork": {"dining area": 1, "kitchen": 1},
    "knife": {"kitchen": 1, "dining area": 1},
    "spoon": {"kitchen": 1, "dining area": 1},
    "wine glass": {"dining area": 1, "kitchen": 1},
    "couch": {"living room": 3},
    "tv": {"living room": 2},
    "remote": {"living room": 1},
    "potted plant": {"living room": 0.5},
    "laptop": {"workspace": 1.5},
    "keyboard": {"workspace": 2},
    "mouse": {"workspace": 2},
    "car": {"outdoor area": 2},
    "truck": {"outdoor area": 2},
    "bus": {"outdoor area": 2},
    "motorcycle": {"outdoor area": 2},
    "bicycle": {"outdoor area": 1.5},
    "traffic light": {"outdoor area": 3},
    "stop sign": {"outdoor area": 3},
    "bench": {"outdoor area": 1.5},
}


def guess_room_from_objects():
    """Returns (room, [supporting objects]) or None if there isn't enough evidence."""
    with recent_lock:
        frames = list(recent_frames)
    if not frames:
        return None

    scores = collections.Counter()
    evidence = collections.defaultdict(collections.Counter)
    for _, names in frames:
        for cls in names:
            for room, weight in ROOM_CLUES.get(cls, {}).items():
                scores[room] += weight
                evidence[room][cls] += weight
    if not scores:
        return None

    room, score = scores.most_common(1)[0]
    if score / len(frames) < SCENE_MIN_SCORE:
        return None
    return room, [cls for cls, _ in evidence[room].most_common(3)]


# Method 2 (optional, more accurate): a Places365 ResNet18 scene classifier
# that looks at the whole camera image and picks from 365 scene types
# (kitchen, bedroom, living room, corridor, street, ...). It only runs when
# someone asks, never every frame, and is loaded the first time it's needed.
_places = {"tried": False, "net": None, "labels": None, "preprocess": None, "torch": None}


def _load_places365():
    if _places["tried"]:
        return _places["net"] is not None
    _places["tried"] = True

    if not (os.path.exists(PLACES_WEIGHTS) and os.path.exists(PLACES_LABELS)):
        print("Places365 files not found - using the object-based scene guess only.")
        return False
    try:
        import torch
        import torchvision.transforms as T
        from torchvision.models import resnet18

        # Keep torch from grabbing every core and starving the YOLO loop.
        torch.set_num_threads(2)

        # The official file comes from the MIT Places365 release; it is a
        # pickle, so newer torch needs weights_only=False to open it.
        checkpoint = torch.load(PLACES_WEIGHTS, map_location="cpu", weights_only=False)
        state = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        net = resnet18(num_classes=365)
        net.load_state_dict(state)
        net.eval()

        labels = []
        with open(PLACES_LABELS) as f:
            for line in f:
                # "/k/kitchen 203" -> "kitchen";  "/a/apartment_building/outdoor 3" -> "apartment building outdoor"
                name = line.strip().split(" ")[0][3:]
                labels.append(name.replace("_", " ").replace("/", " "))

        _places["preprocess"] = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        _places["torch"] = torch
        _places["labels"] = labels
        _places["net"] = net
        print("Places365 scene classifier loaded.")
        return True
    except Exception as e:
        print(f"Couldn't load Places365 ({e}) - using the object-based scene guess only.")
        return False


def classify_scene_places365(frame):
    torch = _places["torch"]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    batch = _places["preprocess"](rgb).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_places["net"](batch), dim=1)[0]
    conf, idx = probs.max(dim=0)
    return _places["labels"][int(idx)], float(conf)


def _with_article(phrase):
    return ("an " if phrase[0] in "aeiou" else "a ") + phrase


def _join_words(items):
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def answer_scene():
    frame = latest_frame
    if frame is None:
        return "I can't see anything yet"

    places_label = None
    if _load_places365():
        try:
            label, conf = classify_scene_places365(frame)
            print(f"Places365: {label} ({conf:.2f})")
            if conf >= PLACES_MIN_CONF:
                return f"This looks like {_with_article(label)}"
            places_label = label
        except Exception as e:
            print(f"Places365 classification failed: {e}")

    guess = guess_room_from_objects()
    if guess:
        room, seen = guess
        return f"This looks like {_with_article(room)}. I can see {_join_words(seen)}"
    if places_label:
        return f"I am not sure, but it might be {_with_article(places_label)}"
    return "I can't tell what kind of place this is"


# ---------- VOICE QUERIES ----------
SCENE_PHRASES = (
    "where am i", "where are we", "what room", "what place", "what scene",
    "what kind of room", "what kind of place", "what kind of area",
    "describe the scene", "describe the scenery", "scenery", "my surroundings",
    "what is around me", "what do you see around me", "describe my surroundings",
    "what is my location", "what environment am i in",
)


def handle_voice_query(text):
    text = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s']", " ", text.lower())).strip()

    # Check specific intents first so questions mentioning multiple topics are
    # not accidentally treated as object-identification requests.
    if re.search(r"\b(weather|temperature|forecast|rain)\b", text):
        speak(answer_weather())
    elif re.search(r"\btime\b", text):
        speak(answer_time())
    elif re.search(r"\b(date|day|today|weekday|tomorrow)\b", text):
        speak(answer_date())
    elif any(phrase in text for phrase in SCENE_PHRASES) or re.search(
        r"\b(room|place|scene|area|surroundings|environment|location)\b", text
    ):
        speak(answer_scene())
    elif any(phrase in text for phrase in (
        "what is it", "what is this", "what do you see", "what can you see",
        "what kind of object", "identify",
    )) or re.search(r"\b(objects?|obstacles?)\b", text):
        if last_close_object:
            speak(f"It looks like a {last_close_object}")
        else:
            speak("Nothing detected right now")
    else:
        speak(
            "I can tell you the time, date, weather, what I see nearby, "
            "or what kind of place this looks like"
        )


def _find_webcam_mic_index():
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        if "webcam" in name.lower():
            return index
    return None


class StereoMicrophone(sr.Microphone):
    """sr.Microphone always opens its capture stream as mono (hardcoded in
    the library), but `arecord --dump-hw-params` shows the webcam mic only
    supports CHANNELS: 2 - it has no mono mode. Opening the raw hw:2,0
    device with a mono request that the hardware can't actually provide is
    what produced garbled audio, even though the same mic works fine
    elsewhere (e.g. a laptop's own audio stack, which auto-converts).
    This overrides the stream to open at the device's real channel count;
    listen_for_query() downmixes the resulting stereo audio to mono before
    handing it to recognize_google()."""

    def __enter__(self):
        assert self.stream is None, "This audio source is already inside a context manager"
        self.audio = self.pyaudio_module.PyAudio()
        try:
            self.stream = sr.Microphone.MicrophoneStream(
                self.audio.open(
                    input_device_index=self.device_index,
                    channels=2,
                    format=self.format,
                    rate=self.SAMPLE_RATE,
                    frames_per_buffer=self.CHUNK,
                    input=True,
                )
            )
        except Exception:
            self.audio.terminate()
            raise
        return self


recognizer = sr.Recognizer()
# The Pi's ALSA "default" input device resolves to the HDMI output (card 0),
# which has no real microphone, so recognize_google() never gets usable
# audio. Select the webcam's USB mic (the only real capture device) directly.
_webcam_mic_index = _find_webcam_mic_index()
_mic_names = sr.Microphone.list_microphone_names()
_selected_name = _mic_names[_webcam_mic_index] if _webcam_mic_index is not None else "<default, webcam not found>"
print(f"Using microphone device index {_webcam_mic_index}: {_selected_name}")
# The device advertises RATE: [8000 48000] but only genuinely captures
# cleanly at 8000 - PyAudio's auto-detected default rate produced sped-up/
# garbled audio (confirmed via arecord -r 16000/48000 vs -r 8000 A/B tests).
mic = StereoMicrophone(device_index=_webcam_mic_index, sample_rate=8000)


def listen_for_query():
    global last_close_object
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
    print("Mic ready, listening for questions...")
    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
            # StereoMicrophone captures real 2-channel audio; downmix to mono
            # before recognition (and before the debug save below).
            mono_data = audioop.tomono(audio.get_raw_data(), audio.sample_width, 0.5, 0.5)
            audio = sr.AudioData(mono_data, audio.sample_rate, audio.sample_width)
            print("Heard something, recognizing...")
            text = recognizer.recognize_google(audio).lower()
            print(f"You said: {text}")
            handle_voice_query(text)
        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            # Temporary diagnostic: save what was actually captured so we can
            # inspect it directly, since the mic hardware itself tests fine
            # elsewhere (e.g. Google Meet) but recognize_google() keeps failing.
            debug_path = f"{RECORDINGS_DIR}/debug_audio_{time.strftime('%Y%m%d_%H%M%S')}.wav"
            try:
                with open(debug_path, "wb") as f:
                    f.write(audio.get_wav_data())
                print(f"Heard audio but couldn't understand it (saved {debug_path})")
            except OSError as e:
                print(f"Heard audio but couldn't understand it (couldn't save debug audio: {e})")
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
    latest_frame = frame  # lets the voice thread look at the current view

    results = model(frame, verbose=False, conf=CONF_THRESHOLD)
    annotated = results[0].plot()

    h, w = frame.shape[:2]
    detected_this_frame = False

    largest_ratio_this_frame = 0
    largest_object_this_frame = None
    largest_box_this_frame = None
    names_this_frame = set()

    for box in results[0].boxes:
        cls = model.names[int(box.cls)]
        x1, y1, x2, y2 = box.xyxy[0]
        width_ratio = (x2 - x1) / w
        detected_this_frame = True
        names_this_frame.add(cls)

        if width_ratio > largest_ratio_this_frame:
            largest_ratio_this_frame = width_ratio
            largest_object_this_frame = cls
            largest_box_this_frame = (x1, x2)

    # Keep a short rolling history of what was seen, for scene identification.
    now_ts = time.time()
    with recent_lock:
        recent_frames.append((now_ts, names_this_frame))
        while recent_frames and recent_frames[0][0] < now_ts - SCENE_WINDOW_SECONDS:
            recent_frames.popleft()

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
