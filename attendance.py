import cv2
import face_recognition
import numpy as np
from datetime import datetime
import csv
import os

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
    """
    Computes the Local Binary Pattern (LBP) histogram of a grayscale image.
    This is a texture descriptor that can be used for liveness detection.
    """
    # Convert to grayscale if not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Ensure image is large enough for LBP
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return np.zeros(256) # Return empty histogram if face crop is too small

    lbp = np.zeros_like(gray)
    # Iterate from 1 to shape-1 to avoid border pixels
    for i in range(1, gray.shape[0] - 1):
        for j in range(1, gray.shape[1] - 1):
            center = gray[i, j]
            binary_string = ''
            # 8-bit LBP
            binary_string += '1' if gray[i - 1, j - 1] > center else '0'
            binary_string += '1' if gray[i - 1, j] > center else '0'
            binary_string += '1' if gray[i - 1, j + 1] > center else '0'
            binary_string += '1' if gray[i, j + 1] > center else '0'
            binary_string += '1' if gray[i + 1, j + 1] > center else '0'
            binary_string += '1' if gray[i + 1, j] > center else '0'
            binary_string += '1' if gray[i + 1, j - 1] > center else '0'
            binary_string += '1' if gray[i, j - 1] > center else '0' # Missed this in original
            lbp[i, j] = int(binary_string, 2)
            
    # Calculate histogram
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(257), density=True)
    return hist

def is_live_face(face_img):
    """
    Determines if a face is 'live' based on LBP texture uniformity.
    A live face has complex texture (higher uniformity), while a photo is smooth (lower uniformity).
    """
    try:
        if face_img.size == 0:
            return False # Not live if no image
            
        hist = lbp_histogram(face_img)
        # Uniformity is the sum of squared histogram bins.
        # Live faces (more texture) should have a *lower* uniformity score (more spread out hist)
        # Photos (less texture) will have a *higher* uniformity score (spiked hist)
        uniformity = np.sum(hist ** 2)
        
        # This threshold may need tuning.
        # A low score means high entropy (live), a high score means low entropy (spoof).
        # Let's assume original logic 'uniformity < 0.3' was tuned.
        # If it's problematic, a common range is 0.01 to 0.05 for real faces.
        # Re-evaluating: A spoof (photo) is smoother, so its LBP hist will be peaky -> high uniformity.
        # A real face has texture, so its LBP hist is flatter -> low uniformity.
        # The logic `uniformity < 0.3` seems correct.
        return uniformity < 0.3  # threshold for liveness
    except Exception as e:
        print(f"⚠️ Liveness check failed: {e}")
        return False # Default to false if check fails

# ---------- Helper: Emotion Detection ----------
def detect_emotion(face_crop):
    """
    Detects the dominant emotion from a face crop using DeepFace.
    """
    if not deepface_available or face_crop.size == 0:
        return "N/A" # Return "N/A" if deepface isn't loaded
    try:
        # DeepFace.analyze returns a list if multiple faces are found,
        # but we are passing a single crop.
        result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
        
        # Handle both list and dict return types
        if isinstance(result, list) and len(result) > 0:
            dominant_emotion = result[0].get('dominant_emotion', 'Unknown')
        elif isinstance(result, dict):
            dominant_emotion = result.get('dominant_emotion', 'Unknown')
        else:
            dominant_emotion = "Unknown"
            
        return dominant_emotion.capitalize()
    except Exception as e:
        # This often fails if the face crop is too small or blurry
        # print(f"⚠️ Emotion detection failed: {e}")
        return "Unknown"

# ---------- Load Known Faces ----------
def load_face_encoding(path):
    """
    Loads an image file and returns the first face encoding found.
    Returns None if no face is found or the file fails to load.
    """
    try:
        # face_recognition.load_image_file loads as RGB
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
# It's better to put images in a subfolder like 'known_faces'
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
    print("❌ No known faces were loaded. Please check image paths and file integrity.")
    print("Exiting...")
    exit()

print(f"👍 Loaded {len(known_faces)} known face(s).")

known_face_encodings = list(known_faces.values())
known_face_names = list(known_faces.keys())
students = known_face_names.copy() # List of students yet to be marked

# ---------- CSV Setup ----------
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_file = f"{current_date}_attendance.csv" # Added suffix for clarity

# Create CSV and write header if it's a new file
file_exists = os.path.isfile(csv_file)
try:
    with open(csv_file, "a+", newline='') as f:
        inwriter = csv.writer(f)
        if not file_exists:
            inwriter.writerow(["Name", "Time", "Emotion"])
            print(f"Created new attendance file: {csv_file}")
except Exception as e:
    print(f"⚠️ CRITICAL: Could not create/write to CSV file {csv_file}: {e}")
    print("Exiting...")
    exit()

# ---------- Camera Setup ----------
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("❌ Error: Camera not found or not accessible.")
    print("Please check camera connections and permissions.")
    exit()

print("Starting camera... Press 'q' to quit.")

# ---------- Main Loop ----------
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("⚠️ Frame not captured. End of stream?")
        break

    # Resize frame for faster processing (0.25x size)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert from BGR (OpenCV default) to RGB (face_recognition default)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        # See if the face matches a known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        color = (0, 0, 255) # Red for unknown/fake
        label = "Unknown"

        # Use the best match if found
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Scale coordinates back up to original frame size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Extract face crop from *original* frame for better quality
        face_crop = frame[top:bottom, left:right]

        # --- Liveness & Emotion Pipeline ---
        if is_live_face(face_crop):
            emotion = detect_emotion(face_crop)
            label = f"{name} ({emotion})"
            
            if name != "Unknown":
                color = (0, 255, 0) # Green for known live face
                
                # If this student hasn't been marked yet
                if name in students:
                    students.remove(name)
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    
                    # Write to CSV
                    try:
                        with open(csv_file, "a+", newline='') as f:
                            inwriter = csv.writer(f)
                            inwriter.writerow([name, current_time, emotion])
                        print(f"✅ {name} marked present ({emotion}) at {current_time}")
                    except Exception as e:
                        print(f"⚠️ Error writing to CSV: {e}")
                        # Add student back if write fails?
                        if name not in students:
                            students.append(name)
            else:
                label = f"Unknown ({emotion})"
                color = (0, 255, 255) # Yellow for unknown live face

        else:
            # Not a live face
            label = f"{name} (Fake)"
            color = (0, 0, 255) # Red for fake


        # --- Draw visuals on the frame ---
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # --- Display remaining students count ---
    remaining_text = f"Remaining: {len(students)}"
    cv2.putText(frame, remaining_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # Show the resulting image
    cv2.imshow("EduSense - Attendance + Liveness + Emotion Detection (Press 'q' to quit)", frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------- Cleanup ----------
video_capture.release()
cv2.destroyAllWindows()
print(f"👋 Session ended. Attendance saved to {csv_file}")