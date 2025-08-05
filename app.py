import streamlit as st
import torch
import os
import cv2
from pathlib import Path
from tempfile import NamedTemporaryFile
from PIL import Image
import subprocess
import sys
import numpy as np

# === Streamlit config ===
st.set_page_config(page_title="Fire Detection App", layout="centered")
st.title("Fire Detection App using YOLOv5")

# === Paths ===
model_path = Path("yolov5/runs/train/exp3/weights/best.pt")
output_dir = "runs/detect"
output_name = "streamlit_exp"

# === Load model ===
@st.cache_resource
def load_model():
    if not model_path.exists():
        st.error(f"Model not found at: {model_path}")
        st.stop()
    return torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path), force_reload=True)

model = load_model()
model.conf = 0.25

# === Fire Alert Logic ===
def fire_level_text(confidences):
    count = len(confidences)
    if count == 0:
        return ""
    high = any(c >= 0.65 for c in confidences)
    mid = any(0.45 <= c < 0.65 for c in confidences)
    low = any(0.25 <= c < 0.45 for c in confidences)

    if count > 2:
        if high:
            return "🚨 High fire detected - Warning! Caution for wildfire."
        elif mid:
            return "⚠️ Mid fire detected - Take caution."
        elif low:
            return "🔥 Fire detected."
    else:
        max_conf = max(confidences)
        if max_conf >= 0.65:
            return "🚨 High fire detected - Warning! Caution for wildfire."
        elif max_conf >= 0.45:
            return "⚠️ Mid fire detected - Take caution."
        elif max_conf >= 0.25:
            return "🔥 Fire detected."
    return ""

# === Uploads ===
uploaded_images = st.file_uploader("Upload up to 15 images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
uploaded_video = st.file_uploader("Or upload a single video", type=["mp4"])

# === IMAGE DETECTION ===
if uploaded_images:
    st.subheader("Image Fire Detection Results")

    if len(uploaded_images) > 15:
        st.warning("You can only upload up to 15 images.")
        uploaded_images = uploaded_images[:15]

    fire_count = 0
    total_images = len(uploaded_images)

    for idx, img_file in enumerate(uploaded_images, 1):
        st.markdown(f"---\n### Image {idx}: {img_file.name}")
        image = Image.open(img_file)
        st.image(image, caption="Original Image", use_container_width=True)

        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            results = model(temp_file.name)
            results.save()

        detect_dirs = sorted(Path("runs/detect").glob("exp*"), key=os.path.getmtime)
        latest_dir = detect_dirs[-1] if detect_dirs else None
        if latest_dir:
            result_images = list(latest_dir.glob("*.jpg"))
            if result_images:
                st.image(str(result_images[0]), caption="Detection Result", use_container_width=True)
                detections = results.pandas().xyxy[0]
                if not detections.empty:
                    confidences = detections['confidence'].tolist()
                    avg_conf = float(np.mean(confidences))
                    st.info(f"Detected {len(confidences)} fire(s) with average confidence: {avg_conf:.2f}")
                    st.warning(fire_level_text(confidences))
                    fire_count += 1
                else:
                    st.success("No fire detected in this image.")
            else:
                st.warning("No result image found.")
        else:
            st.warning("Detection folder not found.")

    percent = (fire_count / total_images) * 100 if total_images else 0
    st.markdown("---")
    st.markdown("### Summary")
    st.markdown(f"- Total images uploaded: **{total_images}**")
    st.markdown(f"- Images with fire detected: **{fire_count}**")
    st.markdown(f"- Fire detection rate: **{percent:.1f}%**")

# === VIDEO DETECTION ===
elif uploaded_video:
    st.subheader("Video Fire Detection")

    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_video.read())

    st.info("Running detection, please wait...")

    output_path = Path(output_dir) / output_name
    if output_path.exists():
        for f in output_path.glob("*.mp4"):
            try:
                f.unlink()
            except:
                pass

    detect_script = Path("yolov5/detect.py").resolve()
    subprocess.run([
        sys.executable, str(detect_script),
        "--weights", str(model_path),
        "--img", "640", "--conf", "0.25",
        "--source", temp_video.name,
        "--project", output_dir, "--name", output_name,
        "--exist-ok"
    ], check=True)

    if output_path.exists():
        video_files = list(output_path.glob("*.mp4"))
        if video_files:
            latest_video = max(video_files, key=os.path.getctime)
            if latest_video.stat().st_size > 0:
                st.success("Fire detection complete!")

                try:
                    # Try to extract video info
                    cap = cv2.VideoCapture(str(latest_video))
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    duration = frame_count / fps if fps > 0 else 0

                    st.markdown(f"**Video Duration**: {duration:.1f} seconds")

                    # Convert to web-friendly format using ffmpeg
                    web_video_path = latest_video.parent / f"web_{latest_video.name}"
                    try:
                        subprocess.run([
                            "ffmpeg", "-i", str(latest_video),
                            "-c:v", "libx264", "-preset", "fast",
                            "-c:a", "aac", "-b:a", "128k",
                            "-movflags", "+faststart",
                            "-y", str(web_video_path)
                        ], check=True, capture_output=True)
                        st.info("Converted to web-compatible format.")
                        st.video(str(web_video_path))
                    except (subprocess.CalledProcessError, FileNotFoundError) as ffmpeg_err:
                        st.warning("FFmpeg conversion failed, showing original video instead.")
                        st.video(str(latest_video))

                    st.markdown("Fire may be present based on detection — please manually review the video.")
                except Exception as e:
                    st.error(f"Failed to read video metadata or convert: {e}")
                    st.video(str(latest_video))
            else:
                st.error("Processed video is empty.")
        else:
            st.warning("No video output found.")
    else:
        st.warning("Detection output folder not created.")

    # Clean up temporary uploaded file
    try:
        os.unlink(temp_video.name)
    except:
        pass

