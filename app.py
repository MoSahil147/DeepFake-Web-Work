
import streamlit as st
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------------------------------------------------
# Paths and folders
# -------------------------------------------------------------------
BASE_PATH     = "/content/drive/MyDrive/DeepFake Datasets"
MODEL_PATH    = os.path.join(BASE_PATH, "deepfake_mlp_model.h5")
UPLOAD_FOLDER = os.path.join(BASE_PATH, "WebMedia", "upload")
FRAME_FOLDER  = os.path.join(BASE_PATH, "WebMedia", "frames")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER,  exist_ok=True)

# -------------------------------------------------------------------
# Load model and feature extractor
# -------------------------------------------------------------------
model = load_model(MODEL_PATH)
feature_extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# -------------------------------------------------------------------
# Streamlit configuration
# -------------------------------------------------------------------
st.set_page_config(page_title="DeepFake Detection", layout="centered")
st.title("DeepFake Detection Tool")

# Resource links under title
col1, col2, col3 = st.columns([5, 3, 3])
with col2:
    st.markdown(
        "[DeepFake Samples for Testing](https://drive.google.com/drive/folders/1J5p4AW3UNRRAVOfmGTYnHW7M8dZBd_Vv?usp=share_link)",
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        "[Real Samples for Testing](https://drive.google.com/drive/folders/1uSYEwZ8RvcTf2eJNwFsH9whntNabEZGF?usp=share_link)",
        unsafe_allow_html=True
    )

st.markdown("Upload a short **.mp4** video for DeepFake analysis.")

# -------------------------------------------------------------------
# File uploader
# -------------------------------------------------------------------
uploaded_file = st.file_uploader("Choose an MP4 video", type=["mp4"])

if uploaded_file is not None:
    video_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(video_path)
    st.info("Video uploaded successfully. Extracting frames...")

    for fname in os.listdir(FRAME_FOLDER):
        os.remove(os.path.join(FRAME_FOLDER, fname))

    cap   = cv2.VideoCapture(video_path)
    fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 1
    count = saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % fps == 0:
            frame_path = os.path.join(FRAME_FOLDER, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1
    cap.release()

    frame_files = sorted(f for f in os.listdir(FRAME_FOLDER) if f.endswith(".jpg"))

    if not frame_files:
        st.error("No frames were extracted. Please try another video.")
    else:
        st.success(f"Extracted {len(frame_files)} frames.")
        st.subheader("Sample Frames")
        cols = st.columns(5)
        for i, frame_name in enumerate(frame_files[:5]):
            img_path = os.path.join(FRAME_FOLDER, frame_name)
            cols[i % 5].image(Image.open(img_path),
                              caption=f"Frame {i + 1}",
                              use_column_width=True)

        st.info("Running DeepFake prediction...")
        features = []
        for frame_name in frame_files[:5]:
            img_path = os.path.join(FRAME_FOLDER, frame_name)
            img = image.load_img(img_path, target_size=(224, 224))
            x   = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
            feat = feature_extractor.predict(x, verbose=0)
            features.append(feat[0])

        avg_feature = np.mean(features, axis=0).reshape(1, -1)
        prediction  = model.predict(avg_feature, verbose=0)[0][0]

        st.subheader("Prediction Result")
        confidence = prediction if prediction > 0.5 else 1 - prediction
        label      = "FAKE" if prediction > 0.5 else "REAL"

        st.markdown(f"Prediction: **{label}**")
        st.markdown(f"Confidence Score: {confidence:.2f}")
else:
    st.warning("Please upload an MP4 video to continue.")
