# scripts/add_user.py

import cv2
import os
import numpy as np
from deepface import DeepFace

IMAGES_DIR = "data/images"
EMBEDDINGS_DIR = "data/embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def get_embedding(image_path):
    try:
        embedding = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]
        return np.array(embedding)
    except Exception as e:
        print(f"[!] Error processing {image_path}: {e}")
        return None

def register_user(user_id, image_paths):
    embeddings = []

    for path in image_paths:
        emb = get_embedding(path)
        if emb is not None:
            embeddings.append(emb)

    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        np.save(os.path.join(EMBEDDINGS_DIR, f"{user_id}.npy"), avg_embedding)
        print(f"[✓] Registered user '{user_id}' with {len(embeddings)} embeddings.")
    else:
        print("[!] No valid embeddings found. User not registered.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("user_id")
    parser.add_argument("images", nargs="+", help="Paths to user's face images")
    args = parser.parse_args()

    register_user(args.user_id, args.images)
