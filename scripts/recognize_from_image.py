import os
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from datetime import datetime
import csv

# Paths
IMAGE_PATH = r"C:\Users\Hp\Desktop\Ai Projects\attendence\scripts\group_photo.jpg"
EMBEDDINGS_DIR = "data/embeddings"
ATTENDANCE_LOG = "attendance/attendance_log.csv"
os.makedirs("attendance", exist_ok=True)

# MediaPipe setup
mp_face_detection = mp.solutions.face_detection

# Load all known face embeddings
def load_known_faces():
    known_faces = {}
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".npy"):
            user_id = os.path.splitext(file)[0]
            embedding = np.load(os.path.join(EMBEDDINGS_DIR, file))
            known_faces[user_id] = embedding
    return known_faces

# Compare face embeddings
def is_match(known_embedding, new_embedding, threshold=10):  # Adjusted threshold
    dist = np.linalg.norm(known_embedding - new_embedding)
    return dist < threshold

# Mark attendance in log
def mark_attendance(user_id):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    already_marked = False
    if os.path.exists(ATTENDANCE_LOG):
        with open(ATTENDANCE_LOG, "r") as f:
            already_marked = any(user_id in line for line in f.readlines())

    if not already_marked:
        with open(ATTENDANCE_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([user_id, timestamp])
        print(f"[✓] Marked attendance for {user_id}")
    else:
        print(f"[•] Already marked for {user_id}")

# Main function
def recognize_from_image():
    known_faces = load_known_faces()
    frame = cv2.imread(IMAGE_PATH)

    if frame is None:
        print(f"[!] ERROR: Could not read image at path: {IMAGE_PATH}")
        return

    with mp_face_detection.FaceDetection(min_detection_confidence=0.6) as detector:
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not results.detections:
            print("[!] No faces detected.")
            return

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            w_box = int(bbox.width * w)
            h_box = int(bbox.height * h)

            # Ensure bounding box is within image
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
                print(f"[!] DeepFace error: {e}")
                continue

            matched = False
            for user_id, known_emb in known_faces.items():
                if is_match(known_emb, np.array(emb)):
                    mark_attendance(user_id)
                    cv2.putText(frame, f"{user_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    matched = True
                    break

            if not matched:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Recognized Faces", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_from_image()
