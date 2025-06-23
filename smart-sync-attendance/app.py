import gradio as gr
import os
import cv2
import numpy as np
from deepface import DeepFace
from datetime import datetime
import csv
from PIL import Image
import tempfile
import pandas as pd

# Directory setup (paths for Hugging Face Spaces)
EMBEDDINGS_DIR = "data/embeddings"
IMAGES_DIR = "data/images"
ATTENDANCE_LOG = "attendance/attendance_log.csv"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs("attendance", exist_ok=True)

# Admin credentials
ADMIN_CREDENTIALS = {
    "admin1": "pass123",
    "admin2": "secure456"
}

# Save uploaded images
def save_uploaded_images(user_id, uploaded_images):
    user_dir = os.path.join(IMAGES_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    saved_paths = []
    for i, img in enumerate(uploaded_images):
        if img is None:
            continue
        img_path = os.path.join(user_dir, f"{i}.jpg")
        # Save PIL image directly
        img.save(img_path)
        saved_paths.append(img_path)
    return saved_paths

# Extract embedding using DeepFace with explicit local model
def get_embedding(image_path):
    try:
        # Force local model usage
        embedding = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False, detector_backend="opencv")[0]["embedding"]
        return np.array(embedding)
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return None

# Register user by averaging embeddings
def register_user(user_id, images):
    if not user_id or not images:
        return "Please provide a valid user ID and at least 1 image."
    embeddings = []
    saved_paths = save_uploaded_images(user_id, images)
    if not saved_paths:
        return "No valid images provided."
    for img_path in saved_paths:
        emb = get_embedding(img_path)
        if emb is not None:
            embeddings.append(emb)
    if embeddings:
        avg_emb = np.mean(embeddings, axis=0)
        np.save(os.path.join(EMBEDDINGS_DIR, f"{user_id}.npy"), avg_emb)
        return f"User '{user_id}' registered successfully!"
    return "Failed to register user due to embedding issues."

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
    return f"Attendance marked for {user_id} on {date_str} at {time_str}."

# Recognize from image (webcam or upload)
def recognize_from_image(image):
    if image is None:
        return "No image provided."
    # Save PIL image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        try:
            emb = get_embedding(temp_file.name)
            if emb is not None:
                known_faces = load_known_faces()
                matched = []
                for user_id, known_emb in known_faces.items():
                    if is_match(known_emb, emb):
                        mark_attendance(user_id)
                        matched.append(user_id)
                return f"Matched: {', '.join(matched) if matched else 'No match found'}"
            return "Error processing image."
        except Exception as e:
            print(f"Error in recognize_from_image: {e}")
            return f"Error: {str(e)}"
        finally:
            os.unlink(temp_file.name)

# View log function
def view_log():
    if os.path.exists(ATTENDANCE_LOG):
        log_df = pd.read_csv(ATTENDANCE_LOG, header=None, names=["User", "Status", "Date", "Time"])
        grouped = log_df.groupby("Date")
        output = []
        for date, group in grouped:
            day_name = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
            output.append(f"### 📅 Date: {date} ({day_name})")
            output.append(group.drop(columns=["Date"]).to_html(index=False))
        return "\n".join(output), log_df.to_csv(index=False).encode('utf-8')
    return "No attendance data yet.", None

# Admin login
def admin_login(username, password):
    if ADMIN_CREDENTIALS.get(username) == password:
        return True, "Login successful!"
    return False, "Invalid credentials."

# Admin logout
def admin_logout():
    return False, "Logged out successfully."

# Admin report summary
def get_admin_summary():
    if os.path.exists(ATTENDANCE_LOG):
        df = pd.read_csv(ATTENDANCE_LOG, header=None, names=["User", "Status", "Date", "Time"])
        attendance_summary = df[df["Status"] == "Present"].groupby("User").count()["Status"]
        total_days = df["Date"].nunique()
        summary_df = pd.DataFrame({
            "Total Days Present": attendance_summary,
            "Attendance %": (attendance_summary / total_days * 100).round(2)
        }).fillna(0)
        return summary_df.to_html(index=False), summary_df.to_csv(index=False).encode('utf-8')
    return "No attendance data available.", None

# Delete user
def delete_user(user_to_delete):
    if not user_to_delete:
        return "Select a user to delete."
    emb_path = os.path.join(EMBEDDINGS_DIR, f"{user_to_delete}.npy")
    img_path = os.path.join(IMAGES_DIR, user_to_delete)
    if os.path.exists(emb_path):
        os.remove(emb_path)
    if os.path.exists(img_path):
        for f in os.listdir(img_path):
            os.remove(os.path.join(img_path, f))
        os.rmdir(img_path)
    return f"User '{user_to_delete}' deleted."

# Gradio interface with custom CSS
with gr.Blocks(css="""
    body {
        background: #1c2526;
        color: #d1d5db;
        font-family: 'Poppins', sans-serif;
    }
    .gr-tab {
        background: #2e3a3d;
        border-radius: 5px;
        padding: 10px;
        margin: 5px;
    }
    .gr-tab-selected {
        background: #00c4cc;
        color: #ffffff;
    }
    .gr-button {
        background: #00c4cc;
        color: #ffffff;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1rem;
        font-weight: 600;
        border: none;
    }
    .gr-button:hover {
        background: #00a3a9;
    }
    .gr-textbox, .gr-image, .gr-dropdown {
        background: #2e3a3d;
        color: #d1d5db;
        border: 1px solid #37464b;
        border-radius: 5px;
        padding: 10px;
    }
    .gr-textbox:focus, .gr-dropdown:focus {
        border-color: #00c4cc;
    }
    h1 {
        color: #ffffff;
        text-align: center;
    }
    h1 span {
        color: #00c4cc;
    }
    .subtitle {
        color: #a0aec0;
        text-align: center;
        font-size: 1.1rem;
    }
""") as demo:
    gr.Markdown("# 🔍 SmartSync <span>Attendance</span>\n<p class='subtitle'>Precision-Powered Presence</p>")

    with gr.Tabs():
        with gr.TabItem("📝 Register User"):
            user_id = gr.Textbox(label="Enter User ID (e.g., Name or Roll Number)", placeholder="Enter user ID...")
            images = gr.File(label="Upload 3+ Clear Face Images", file_types=["image"], file_count="multiple")
            images_converted = gr.State()
            register_btn = gr.Button("Register User")
            register_output = gr.Textbox(label="Result")

            def convert_images(files):
                return [Image.open(f.name) for f in files] if files else []

            images.change(fn=convert_images, inputs=images, outputs=images_converted)
            register_btn.click(fn=register_user, inputs=[user_id, images_converted], outputs=register_output)

        with gr.TabItem("✅ Mark Attendance"):
            webcam_input = gr.Image(type="pil", label="Capture Face via Webcam")
            output = gr.Textbox(label="Result")
            submit_btn = gr.Button("Submit")
            submit_btn.click(fn=recognize_from_image, inputs=webcam_input, outputs=output)

        with gr.TabItem("📋 View Log"):
            log_output, log_csv = view_log()
            gr.HTML(log_output)
            if log_csv:
                gr.File(label="Download CSV", value=log_csv, name="attendance_log.csv")

        with gr.TabItem("🔐 Admin Dashboard"):
            admin_state = gr.State(value=False)
            with gr.Row():
                with gr.Column(visible=not admin_state) as login_col:
                    username = gr.Textbox(label="Admin ID", placeholder="Enter Admin ID")
                    password = gr.Textbox(label="Password", type="password", placeholder="Enter Password")
                    login_btn = gr.Button("Login")
                    login_output = gr.Textbox(label="Result")
                with gr.Column(visible=admin_state) as admin_col:
                    logout_btn = gr.Button("Logout")
                    logout_output = gr.Textbox(label="Result")

                    summary_output, summary_csv = get_admin_summary()
                    gr.HTML(summary_output)
                    if summary_csv:
                        gr.File(label="Download Summary", value=summary_csv, name="attendance_summary.csv")

                    new_user_id = gr.Textbox(label="New User ID")
                    new_user_imgs = gr.File(label="Upload 3+ face images", file_types=["image"], file_count="multiple")
                    new_user_imgs_converted = gr.State()
                    admin_register_btn = gr.Button("Register User (Admin)")
                    admin_output = gr.Textbox(label="Result")

                    users = [f.replace(".npy", "") for f in os.listdir(EMBEDDINGS_DIR)]
                    user_to_delete = gr.Dropdown(choices=users, label="Select User to Delete")
                    delete_btn = gr.Button("Delete User")
                    delete_output = gr.Textbox(label="Result")

            login_btn.click(
                fn=admin_login,
                inputs=[username, password],
                outputs=[admin_state, login_output]
            ).then(
                fn=lambda admin_state: (not admin_state, admin_state),
                inputs=admin_state,
                outputs=[login_col, admin_col]
            )

            logout_btn.click(
                fn=admin_logout,
                outputs=[admin_state, logout_output]
            ).then(
                fn=lambda admin_state: (not admin_state, admin_state),
                inputs=admin_state,
                outputs=[login_col, admin_col]
            )

            new_user_imgs.change(fn=convert_images, inputs=new_user_imgs, outputs=new_user_imgs_converted)
            admin_register_btn.click(fn=register_user, inputs=[new_user_id, new_user_imgs_converted], outputs=admin_output)
            delete_btn.click(fn=delete_user, inputs=user_to_delete, outputs=delete_output)

demo.launch()