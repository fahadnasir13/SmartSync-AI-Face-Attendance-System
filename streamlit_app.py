import streamlit as st
import os
import cv2
import numpy as np
from deepface import DeepFace
from datetime import datetime
import csv
from PIL import Image
import tempfile
import pandas as pd

# Directory setup with absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data/embeddings")
IMAGES_DIR = os.path.join(BASE_DIR, "data/images")
ATTENDANCE_LOG = os.path.join(BASE_DIR, "attendance/attendance_log.csv")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(ATTENDANCE_LOG), exist_ok=True)

# Admin credentials
ADMIN_CREDENTIALS = {"admin1": "pass123", "admin2": "secure456"}

# Save uploaded images
def save_uploaded_images(user_id, uploaded_images):
    user_dir = os.path.join(IMAGES_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    saved_paths = []
    for i, img in enumerate(uploaded_images):
        img_path = os.path.join(user_dir, f"{i}.jpg")
        with open(img_path, "wb") as f:
            f.write(img.getbuffer())
        saved_paths.append(img_path)
    return saved_paths

# Extract embedding using DeepFace
def get_embedding(image_path):
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        return np.array(embedding)
    except:
        return None

# Register user by averaging embeddings
def register_user(user_id, image_paths):
    embeddings = []
    for img in image_paths:
        emb = get_embedding(img)
        if emb is not None:
            embeddings.append(emb)
    if embeddings:
        avg_emb = np.mean(embeddings, axis=0)
        np.save(os.path.join(EMBEDDINGS_DIR, f"{user_id}.npy"), avg_emb)
        return True
    return False

# Load known face embeddings
def load_known_faces():
    known_faces = {}
    for file in os.listdir(EMBEDDINGS_DIR):
        user_id = file.replace(".npy", "")
        emb = np.load(os.path.join(EMBEDDINGS_DIR, file))
        known_faces[user_id] = emb
    return known_faces

# Face matching function
def is_match(known_emb, test_emb, threshold=0.6):
    dist = np.linalg.norm(known_emb - test_emb)
    return dist < threshold

# Mark attendance if not already marked today
def mark_attendance(user_id):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    already_marked = False
    if os.path.exists(ATTENDANCE_LOG):
        with open(ATTENDANCE_LOG, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 4 and row[0] == user_id and row[2] == date_str:
                    already_marked = True
                    break
    if not already_marked:
        with open(ATTENDANCE_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([user_id, "Present", date_str, time_str])
        st.success(f"Attendance marked for {user_id} at {time_str}")
    else:
        st.warning(f"Attendance already marked for {user_id} today.")

# Recognize from uploaded image with improved preprocessing
def recognize_from_uploaded_image(image):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(image.getbuffer())
    temp_file.close()
    try:
        # Load and preprocess image
        img = cv2.imread(temp_file.name)
        if img is None:
            st.error("Failed to load the uploaded image.")
            return []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Verify face detection
        result = DeepFace.extract_faces(img_path=temp_file.name, enforce_detection=False)
        if not result:
            st.warning("No face detected in the uploaded image.")
            return []
        emb = DeepFace.represent(img_path=temp_file.name, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        emb = np.array(emb)
        known_faces = load_known_faces()
        matched = []
        for user_id, known_emb in known_faces.items():
            if is_match(known_emb, emb):
                mark_attendance(user_id)
                matched.append(user_id)
        os.unlink(temp_file.name)
        return matched
    except Exception as e:
        st.error(f"Error in face recognition: {str(e)}")
        return []

# Streamlit interface
st.set_page_config(page_title="SmartSync Attendance", layout="centered")

# Apply minimalistic styling
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        body { background: #1c2526; color: #d1d5db; font-family: 'Poppins', sans-serif; }
        .stApp { background: transparent; color: #d1d5db; padding: 15px; max-width: 850px; margin: 0 auto; }
        h1 { color: #ffffff; font-size: 2.5rem; font-weight: 700; text-align: center; margin-bottom: 0.5rem; }
        h1 span { color: #00c4cc; }
        .subtitle { color: #a0aec0; font-size: 1.1rem; text-align: center; margin-bottom: 2rem; font-weight: 400; }
        .stTabs [role="tablist"] { display: flex; justify-content: center; margin-bottom: 1.5rem; background: transparent; }
        .stTabs [role="tab"] { font-size: 1rem; font-weight: 600; color: #a0aec0; padding: 10px 20px; margin: 0 3px; border-radius: 5px; background: #2e3a3d; transition: all 0.3s ease; }
        .stTabs [role="tab"][aria-selected="true"] { background: #00c4cc; color: #ffffff; }
        .stTabs [role="tab"]:hover { background: #37464b; color: #d1d5db; }
        h2 { color: #ffffff; font-size: 1.7rem; font-weight: 600; margin-bottom: 1rem; }
        .stTextInput > div > div > input { background: #2e3a3d; color: #d1d5db; border: 1px solid #37464b; border-radius: 5px; padding: 10px; font-size: 1rem; }
        .stTextInput > div > div > input:focus { border-color: #00c4cc; }
        .stTextInput > label { color: #a0aec0; font-size: 1rem; font-weight: 500; }
        .stButton > button { background: #00c4cc; color: #ffffff; border-radius: 5px; padding: 10px 20px; font-size: 1rem; font-weight: 600; border: none; transition: all 0.3s ease; }
        .stButton > button:hover { background: #00a3a9; }
        .stFileUploader > label { color: #a0aec0; font-size: 1rem; font-weight: 500; }
        .stFileUploader > div { background: #2e3a3d; border: 1px dashed #37464b; border-radius: 5px; padding: 10px; }
        .stTable th, .stTable td { color: #d1d5db; padding: 10px; text-align: center; font-size: 0.95rem; border-bottom: 1px solid #37464b; }
        .stTable th { background: #2e3a3d; font-weight: 600; }
        .stSelectbox select { background: #2e3a3d; color: #d1d5db; border: 1px solid #37464b; border-radius: 5px; padding: 10px; font-size: 1rem; }
        .stAlert { border-radius: 5px; background: #2e3a3d; color: #d1d5db; border: 1px solid #37464b; }
        .stAlert > div { color: #d1d5db; }
        .stSuccess { background: rgba(0, 196, 204, 0.1); border: 1px solid #00c4cc; }
        .stWarning { background: rgba(245, 158, 11, 0.1); border: 1px solid #f59e0b; }
        .stError { background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; }
        .card { background: #2e3a3d; border-radius: 10px; padding: 20px; margin-bottom: 1.5rem; border: 1px solid #37464b; }
    </style>
    """, unsafe_allow_html=True)

# Title and subtitle
st.markdown(
    """
    <h1>🔍 SmartSync <span>Attendance</span></h1>
    <p class="subtitle">Precision-Powered Presence</p>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Register User", "✅ Mark Attendance", "📋 View Log", "🔐 Admin Dashboard"])

# Register User Tab
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Register a New User")
    user_id = st.text_input("Enter User ID (e.g., Name or Roll Number)", placeholder="Enter user ID...")
    uploaded_images = st.file_uploader("Upload 3+ Clear Face Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if st.button("Register User"):
        if user_id and uploaded_images and len(uploaded_images) >= 3:
            paths = save_uploaded_images(user_id, uploaded_images)
            success = register_user(user_id, paths)
            if success:
                st.success(f"User '{user_id}' registered successfully!")
            else:
                st.error("Failed to register user.")
        else:
            st.warning("Please provide a valid user ID and at least 3 images.")
    st.markdown('</div>', unsafe_allow_html=True)

# Mark Attendance Tab
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Mark Attendance")
    uploaded_image = st.file_uploader("Upload a Group Photo", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        results = recognize_from_uploaded_image(uploaded_image)
        if results:
            for user in results:
                st.success(f"✅ Attendance marked for: {user}")
        else:
            st.warning("No matching face found.")
    st.markdown('</div>', unsafe_allow_html=True)

# View Log Tab
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Attendance Log")
    if os.path.exists(ATTENDANCE_LOG):
        log_df = pd.read_csv(ATTENDANCE_LOG, header=None, names=["User", "Status", "Date", "Time"])
        grouped = log_df.groupby("Date")
        full_csv_content = ""
        for date, group in grouped:
            day_name = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
            st.markdown(f"### 📅 Date: {date} ({day_name})")
            styled = group.drop(columns=["Date"]).style.applymap(
                lambda val: 'color: #00c4cc;' if val == 'Present' else 'color: #ef4444;', subset=['Status'])
            st.table(styled)
            full_csv_content += f"# Date: {date} ({day_name})\n"
            full_csv_content += group.to_csv(index=False, header=False)
            full_csv_content += "\n"
        st.download_button("⬇️ Download CSV", full_csv_content.encode('utf-8'), file_name="attendance_log.csv", mime="text/csv")
    else:
        st.info("No attendance data yet.")
    st.markdown('</div>', unsafe_allow_html=True)

# Admin Dashboard Tab
with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔐 Admin Dashboard")
    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False
    if not st.session_state.admin_logged_in:
        username = st.text_input("Admin ID", placeholder="Enter Admin ID")
        password = st.text_input("Password", type="password", placeholder="Enter Password")
        if st.button("Login"):
            if ADMIN_CREDENTIALS.get(username) == password:
                st.session_state.admin_logged_in = True
                st.success("Login successful!")
            else:
                st.error("Invalid credentials.")
    else:
        if st.button("Logout"):
            st.session_state.admin_logged_in = False
            st.experimental_rerun()
        st.subheader("📊 Attendance Report Summary")
        if os.path.exists(ATTENDANCE_LOG):
            df = pd.read_csv(ATTENDANCE_LOG, header=None, names=["User", "Status", "Date", "Time"])
            attendance_summary = df[df["Status"] == "Present"].groupby("User").count()["Status"]
            total_days = df["Date"].nunique()
            summary_df = pd.DataFrame({
                "Total Days Present": attendance_summary,
                "Attendance %": (attendance_summary / total_days * 100).round(2)})
            def highlight_low(val):
                return 'color: #ef4444;' if val < 50 else 'color: #00c4cc;'
            styled = summary_df.style.applymap(highlight_low, subset=["Attendance %"])
            st.table(styled)
            csv_report = summary_df.to_csv().encode("utf-8")
            st.download_button("📥 Download Attendance Summary", csv_report, "attendance_summary.csv", "text/csv")
        else:
            st.info("No attendance data available.")
        st.subheader("➕ Add New User")
        new_user_id = st.text_input("New User ID")
        new_user_imgs = st.file_uploader("Upload 3+ face images", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="admin_upload")
        if st.button("Register User (Admin)"):
            if new_user_id and new_user_imgs and len(new_user_imgs) >= 3:
                paths = save_uploaded_images(new_user_id, new_user_imgs)
                success = register_user(new_user_id, paths)
                if success:
                    st.success(f"User '{new_user_id}' registered successfully.")
                else:
                    st.error("Failed to register user.")
            else:
                st.warning("Provide valid ID and 3+ images.")
        st.subheader("🗑️ Delete Existing User")
        users = [f.replace(".npy", "") for f in os.listdir(EMBEDDINGS_DIR)]
        user_to_delete = st.selectbox("Select User to Delete", users)
        if st.button("Delete User"):
            emb_path = os.path.join(EMBEDDINGS_DIR, f"{user_to_delete}.npy")
            img_path = os.path.join(IMAGES_DIR, user_to_delete)
            if os.path.exists(emb_path):
                os.remove(emb_path)
            if os.path.exists(img_path):
                for f in os.listdir(img_path):
                    os.remove(os.path.join(img_path, f))
                os.rmdir(img_path)
            st.success(f"User '{user_to_delete}' deleted.")
    st.markdown('</div>', unsafe_allow_html=True)