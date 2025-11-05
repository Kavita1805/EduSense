import cv2
import face_recognition
import numpy as np
from datetime import datetime
import csv
import os

# Function: LBP histogram features
def lbp_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = np.zeros_like(gray)
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            center = gray[i, j]
            binary = ''
            binary += '1' if gray[i-1, j-1] > center else '0'
            binary += '1' if gray[i-1, j] > center else '0'
            binary += '1' if gray[i-1, j+1] > center else '0'
            binary += '1' if gray[i, j+1] > center else '0'
            binary += '1' if gray[i+1, j+1] > center else '0'
            binary += '1' if gray[i+1, j] > center else '0'
            binary += '1' if gray[i+1, j-1] > center else '0'
            lbp[i, j] = int(binary, 2)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(257), density=True)
    return hist

# Compare texture score (simple heuristic)
def is_live_face(face_img):
    hist = lbp_histogram(face_img)
    uniformity = np.sum(hist**2)  # high = flat surface (photo), low = natural texture
    return uniformity < 0.3  # threshold (tune this with testing)

# Load known encodings
jane_image = face_recognition.load_image_file("jane.jpg")
jane_encoding = face_recognition.face_encodings(jane_image)[0]
kavitha_image = face_recognition.load_image_file("kavitha.jpg")
kavitha_encoding = face_recognition.face_encodings(kavitha_image)[0]
neha_image = face_recognition.load_image_file("neha.jpg")
neha_encoding = face_recognition.face_encodings(neha_image)[0]
taha_image = face_recognition.load_image_file("taha.jpg")
taha_encoding = face_recognition.face_encodings(taha_image)[0]
rahul_image = face_recognition.load_image_file("rahul.jpg")
rahul_encoding = face_recognition.face_encodings(rahul_image)[0]
shreyas_image = face_recognition.load_image_file("Shreyas .jpg")
shreyas_encoding = face_recognition.face_encodings(shreyas_image)[0]
kaashif_image = face_recognition.load_image_file("kaashif.jpg")
kaashif_encoding = face_recognition.face_encodings(kaashif_image)[0]
naggie_image = face_recognition.load_image_file("naggie.jpg")
naggie_encoding = face_recognition.face_encodings(naggie_image)[0]


known_face_encodings = [jane_encoding, kavitha_encoding, neha_encoding, taha_encoding,rahul_encoding,shreyas_encoding,kaashif_encoding,naggie_encoding]
known_faces_names = ["Jane", "Kavitha", "Neha", "Taha","rahul","Shreyas","kaashif","naggie"]
students = known_faces_names.copy()

# CSV
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_file = f"{current_date}.csv"
file_exists = os.path.isfile(csv_file)
with open(csv_file, "a+", newline='') as f:
    inwriter = csv.writer(f)
    if not file_exists:
        inwriter.writerow(["Name", "Time"])

# Start video
video_capture = cv2.VideoCapture(0)
#address="https://192.168.1.16:8080/video" 
#video_capture.open(address)
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)
        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        # Scale back coords
        top *= 4; right *= 4; bottom *= 4; left *= 4

        # Crop face for liveness test
        face_crop = frame[top:bottom, left:right]
        live = is_live_face(face_crop) if face_crop.size > 0 else False

        if live:
            if name in students and name != "Unknown":
                students.remove(name)
                print(f"✅ Live Attendance Marked: {name}")
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                with open(csv_file, "a+", newline='') as f:
                    inwriter = csv.writer(f)
                    inwriter.writerow([name, current_time])
            label = f"{name} (Live)"
            color = (0, 255, 0)
        else:
            label = f"{name} (Fake)"
            color = (0, 0, 255)

        # Draw
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, label, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Remaining counter
    cv2.putText(frame, f"Remaining: {len(students)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Attendance + Liveness", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break       

video_capture.release()
cv2.destroyAllWindows()
