import os
import numpy as np
from deepface import DeepFace

# Define directories
IMAGES_DIR = "data/images"
EMBEDDINGS_DIR = "data/embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Get embedding using DeepFace with consistent settings
def get_embedding(image_path):
    try:
        print(f"[•] Getting embedding for: {image_path}")
        embedding = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",           # Must match runtime
            enforce_detection=False         # Important to keep consistent
        )[0]["embedding"]
        return np.array(embedding)
    except Exception as e:
        print(f"[!] Error processing {image_path}: {e}")
        return None

# Process each user’s folder
def process_user_folder(user_folder_path):
    user_id = os.path.basename(user_folder_path)
    print(f"[→] Processing user: {user_id}")
    embeddings = []

    for filename in os.listdir(user_folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(user_folder_path, filename)
            if os.path.exists(image_path):
                embedding = get_embedding(image_path)
                if embedding is not None:
                    embeddings.append(embedding)
            else:
                print(f"[!] Image not found: {image_path}")

    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        output_path = os.path.join(EMBEDDINGS_DIR, f"{user_id}.npy")
        np.save(output_path, avg_embedding)
        print(f"[✓] Saved embedding to {output_path}")
    else:
        print(f"[!] No valid embeddings for {user_id}, skipping.")

# Register all users
def register_all_users():
    for user_folder in os.listdir(IMAGES_DIR):
        full_path = os.path.join(IMAGES_DIR, user_folder)
        if os.path.isdir(full_path):
            process_user_folder(full_path)
        else:
            print(f"[!] Skipping non-directory: {full_path}")

if __name__ == "__main__":
    register_all_users()
