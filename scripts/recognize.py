import cv2
import os
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from datetime import datetime
import csv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

EMBEDDINGS_DIR = "data/embeddings"
ATTENDANCE_LOG = "attendance/attendance_log.csv"
os.makedirs("attendance", exist_ok=True)

mp_face_detection = mp.solutions.face_detection

# Load stored user embeddings
def load_known_faces():
    known_faces = {}
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".npy"):
            user_id = os.path.splitext(file)[0]
            embedding = np.load(os.path.join(EMBEDDINGS_DIR, file))
            known_faces[user_id] = embedding
    logging.info(f"[✓] Loaded {len(known_faces)} known faces.")
    return known_faces

# Compare using Euclidean distance
def is_match(known_embedding, new_embedding, threshold=10):
    dist = np.linalg.norm(known_embedding - new_embedding)
    return dist < threshold

# Check if already logged today
def already_marked_today(user_id):
    today = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(ATTENDANCE_LOG):
        return False
    with open(ATTENDANCE_LOG, "r") as f:
        for line in f:
            if user_id in line and today in line:
                return True
    return False

# Write attendance
def write_attendance(user_id, status="Present"):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    row = [user_id, status, date_str, time_str]

    write_header = not os.path.exists(ATTENDANCE_LOG)
    with open(ATTENDANCE_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["User ID", "Status", "Date", "Time"])
        writer.writerow(row)
        logging.info(f"[✓] {status} marked for {user_id}")

def recognize_faces_from_camera():
    known_faces = load_known_faces()
    cap = cv2.VideoCapture(0)

    detected_users = set()

    with mp_face_detection.FaceDetection(min_detection_confidence=0.6) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("[!] Failed to read frame.")
                break

            results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    w_box = int(bbox.width * w)
                    h_box = int(bbox.height * h)

                    x, y = max(0, x), max(0, y)
                    x2, y2 = min(w, x + w_box), min(h, y + h_box)
                    face_crop = frame[y:y2, x:x2]

                    if face_crop.size == 0:
                        continue

                    temp_path = "temp_face.jpg"
                    cv2.imwrite(temp_path, face_crop)

                    try:
                        emb = DeepFace.represent(img_path=temp_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                    except Exception as e:
                        logging.warning(f"[!] Embedding error: {e}")
                        continue

                    matched_user = None
                    for user_id, known_emb in known_faces.items():
                        if is_match(known_emb, np.array(emb)):
                            matched_user = user_id
                            if user_id not in detected_users and not already_marked_today(user_id):
                                write_attendance(user_id, "Present")
                                detected_users.add(user_id)
                            break

                    label = matched_user if matched_user else "Unknown"
                    color = (0, 255, 0) if matched_user else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Live Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Mark others absent
        for user_id in known_faces:
            if user_id not in detected_users and not already_marked_today(user_id):
                write_attendance(user_id, "Absent")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces_from_camera()
