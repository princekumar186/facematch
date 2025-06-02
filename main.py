import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime
import pyttsx3

# === TTS Setup ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    print(f"[TTS] {text}")
    engine.say(text)
    engine.runAndWait()

# === Excel Setup ===
EXCEL_FILE = "attendance.xlsx"

def init_excel():
    if not os.path.exists(EXCEL_FILE):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_excel(EXCEL_FILE, index=False, engine='openpyxl')

def mark_attendance_excel(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    df = pd.read_excel(EXCEL_FILE, engine='openpyxl')

    # Avoid duplicate entries for the same day
    if not ((df["Name"] == name) & (df["Date"] == date)).any():
        new_entry = pd.DataFrame([[name, date, time]], columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_excel(EXCEL_FILE, index=False, engine='openpyxl')
        print(f"[INFO] Marked attendance for {name} at {time}")
        speak(f"Attendance marked for {name}")
    else:
        print(f"[INFO] {name} already marked today.")

# === Load known faces ===
known_faces_dir = 'known_faces'
known_encodings = []
known_names = []

for file in os.listdir(known_faces_dir):
    if file.lower().endswith(('.jpg', '.png')):
        path = os.path.join(known_faces_dir, file)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file)[0])
        else:
            print(f"[WARNING] No face found in {file}")

# === Run Live Attendance ===
def main():
    init_excel()
    cap = cv2.VideoCapture(0)
    print("[INFO] Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("[ERROR] Webcam not detected.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if matches and any(matches):
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = known_names[best_match_index]

                    # Draw face box
                    top, right, bottom, left = [v * 4 for v in face_location]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left + 6, bottom - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Mark attendance
                    mark_attendance_excel(name)

        cv2.imshow("Live Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
