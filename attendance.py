import cv2
import face_recognition
import numpy as np
from datetime import datetime, timedelta
import csv
import os
from collections import defaultdict, deque, Counter

# Try importing DeepFace safely
try:
    from deepface import DeepFace
    deepface_available = True
    print("✅ DeepFace is available for emotion detection.")
except Exception as e:
    print(f"⚠️ DeepFace not available for emotion detection: {e}")
    print("   Emotion detection will be skipped.")
    deepface_available = False

# ---------- Helper: Liveness Detection ----------
def lbp_histogram(image):
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return np.zeros(256)

    lbp = np.zeros_like(gray)
    for i in range(1, gray.shape[0] - 1):
        for j in range(1, gray.shape[1] - 1):
            center = gray[i, j]
            binary_string = ''
            binary_string += '1' if gray[i - 1, j - 1] > center else '0'
            binary_string += '1' if gray[i - 1, j] > center else '0'
            binary_string += '1' if gray[i - 1, j + 1] > center else '0'
            binary_string += '1' if gray[i, j + 1] > center else '0'
            binary_string += '1' if gray[i + 1, j + 1] > center else '0'
            binary_string += '1' if gray[i + 1, j] > center else '0'
            binary_string += '1' if gray[i + 1, j - 1] > center else '0'
            binary_string += '1' if gray[i, j - 1] > center else '0'
            lbp[i, j] = int(binary_string, 2)
            
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(257), density=True)
    return hist

def is_live_face(face_img):
    try:
        if face_img.size == 0:
            return False
        hist = lbp_histogram(face_img)
        uniformity = np.sum(hist ** 2)
        return uniformity < 0.3
    except Exception as e:
        print(f"⚠️ Liveness check failed: {e}")
        return False

# ---------- Helper: Emotion Detection ----------
def detect_emotion(face_crop):
    if not deepface_available or face_crop.size == 0:
        return "N/A"
    try:
        result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list) and len(result) > 0:
            dominant_emotion = result[0].get('dominant_emotion', 'Unknown')
        elif isinstance(result, dict):
            dominant_emotion = result.get('dominant_emotion', 'Unknown')
        else:
            dominant_emotion = "Unknown"
        return dominant_emotion.capitalize()
    except Exception as e:
        return "Unknown"

# ---------- Helper: Emotion Smoothing ----------
def get_stable_emotion(history_deque):
    if not history_deque:
        return "N/A"
    try:
        return Counter(history_deque).most_common(1)[0][0]
    except IndexError:
        return "N/A"

# ---------- Load Known Faces ----------
def load_face_encoding(path):
    try:
        img = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            print(f"✅ Loaded encoding for {path}")
            return encodings[0]
        else:
            print(f"⚠️ No face found in {path}")
            return None
    except Exception as e:
        print(f"❌ Could not load {path}: {e}")
        return None

# --- Define paths to known faces ---
known_face_paths = {
    "Jane": "jane.jpg",
    "Kavitha": "kavitha.jpg",
    "Neha": "neha.jpg",
    "Taha": "taha.jpg",
    "Rahul": "rahul.jpg",
    "Shreyas": "Shreyas .jpg",
    "Kaashif": "kaashif.jpg",
    "Naggie": "naggie.jpg",
}

print("Loading known faces...")
known_faces = {}
for name, path in known_face_paths.items():
    encoding = load_face_encoding(path)
    if encoding is not None:
        known_faces[name] = encoding

if not known_faces:
    print("❌ No known faces were loaded. Please check image paths.")
    exit()

print(f"👍 Loaded {len(known_faces)} known face(s).")

known_face_encodings = list(known_faces.values())
known_face_names = list(known_faces.keys())
students = known_face_names.copy()

# ---------- Emotion Smoothing Setup ----------
EMOTION_HISTORY_SIZE = 15
emotion_histories = defaultdict(lambda: deque(maxlen=EMOTION_HISTORY_SIZE))

# ---------- CSV Setup ----------
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_file = f"{current_date}_attendance.csv"

file_exists = os.path.isfile(csv_file)
try:
    with open(csv_file, "a+", newline='') as f:
        inwriter = csv.writer(f)
        if not file_exists:
            inwriter.writerow(["Name", "Time", "Emotion"])
            print(f"Created new attendance file: {csv_file}")
except Exception as e:
    print(f"⚠️ CRITICAL: Could not create/write to CSV file {csv_file}: {e}")
    exit()

# ---------- Camera Setup ----------
print("Attempting to open camera (index 0)...")
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("⚠️ Camera 0 failed. Trying camera (index 1)...")
    video_capture = cv2.VideoCapture(1)
    if not video_capture.isOpened():
        print("❌ Error: Cannot open camera at index 0 or 1.")
        print("Please check camera connections and permissions.")
        exit()

print("✅ Camera opened successfully. Starting camera... Press 'q' to quit.")

# ---------- Alert Setup ----------
ALERT_DURATION = timedelta(seconds=5)
alert_message = ""
alert_timestamp = None

# --- ⭐ NEW: Cooldown for Sad Logs ---
# This prevents flooding the CSV. We'll only log "Sad" once every 5 minutes per student.
SAD_LOG_COOLDOWN = timedelta(minutes=5)
last_sad_log_time = {}
# ----------------------------------

# ---------- Main Loop ----------
while True:
    # Check if alert should be cleared
    if alert_message and (datetime.now() - alert_timestamp) > ALERT_DURATION:
        alert_message = ""
        alert_timestamp = None

    ret, frame = video_capture.read()
    if not ret:
        print("⚠️ Frame not captured.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        color = (0, 0, 255) # Red
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        face_crop = frame[top:bottom, left:right]

        stable_emotion = "N/A"
        
        if is_live_face(face_crop):
            raw_emotion = detect_emotion(face_crop)
            if name != "Unknown":
                emotion_histories[name].append(raw_emotion)
                stable_emotion = get_stable_emotion(emotion_histories[name])
            else:
                stable_emotion = raw_emotion
            
            label = f"{name} ({stable_emotion})"
            
            # --- ⭐ MODIFIED LOGIC: Separated "Present" from "Sad Alert" ---
            if name != "Unknown":
                color = (0, 255, 0) # Green for known live face
                
                # --- Block 1: Mark Present (Runs ONCE per student) ---
                if name in students:
                    students.remove(name)
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    try:
                        with open(csv_file, "a+", newline='') as f:
                            inwriter = csv.writer(f)
                            inwriter.writerow([name, current_time, stable_emotion])
                        print(f"✅ {name} marked present ({stable_emotion}) at {current_time}")
                    except Exception as e:
                        print(f"⚠️ Error writing to CSV: {e}")
                        if name not in students:
                            students.append(name) # Add back if write failed
                
                # --- Block 2: Check for Sad Alert (Runs CONTINUOUSLY) ---
                if stable_emotion == "Sad":
                    now = datetime.now()
                    # Check if we've ever logged this student as sad,
                    # or if 5 minutes have passed since the last log
                    if name not in last_sad_log_time or (now - last_sad_log_time[name]) > SAD_LOG_COOLDOWN:
                        
                        # 1. Set the on-screen alert
                        alert_message = f"ALERT: Check on {name} (Detected as Sad)"
                        alert_timestamp = now
                        
                        # 2. Log this "Sad" event to the CSV
                        current_time = now.strftime("%H:%M:%S")
                        try:
                            with open(csv_file, "a+", newline='') as f:
                                inwriter = csv.writer(f)
                                # Write a NEW line for this sad event
                                inwriter.writerow([name, current_time, "Sad (Alert)"])
                            print(f"🔔 ALERT: {name} detected as 'Sad'. Logged to CSV.")
                            
                            # 3. Update the timestamp to start the 5-min cooldown
                            last_sad_log_time[name] = now
                            
                        except Exception as e:
                            print(f"⚠️ Error writing SAD ALERT to CSV: {e}")
            # --- ⭐ END OF MODIFIED LOGIC ---
            
            else:
                label = f"Unknown ({stable_emotion})"
                color = (0, 255, 255) # Yellow for unknown live face
        else:
            label = f"{name} (Fake)"
            color = (0, 0, 255) # Red

        # Draw visuals
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # --- Display remaining students count ---
    remaining_text = f"Remaining: {len(students)}"
    cv2.putText(frame, remaining_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # --- Draw the alert banner if active ---
    if alert_message:
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 255), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, alert_message, (30, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the resulting image
    cv2.imshow("EduSense - Attendance + Liveness + Emotion Detection (Press 'q' to quit)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------- Cleanup ----------
video_capture.release()
cv2.destroyAllWindows()
print(f"👋 Session ended. Attendance saved to {csv_file}")