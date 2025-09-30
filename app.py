import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import tempfile

# Page configuration
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 0 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .subtitle {
        color: white;
        text-align: center;
        font-size: 1.3rem;
        margin-top: 0.5rem;
        margin-bottom: 2rem;
        opacity: 0.95;
    }
    .stats-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
    }
    .stats-number {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .stats-label {
        color: #666;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    .video-container {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .info-box {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .mask-icon {
        font-size: 4rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_models():
    face_net = cv2.dnn.readNet(
        "face_detector/deploy.prototxt",
        "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    )
    mask_net = load_model("mask_detector.model.h5")
    return face_net, mask_net

# Face detection and mask prediction
def detect_and_predict_mask(frame, face_net, mask_net):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()
    
    faces = []
    locations = []
    predictions = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]

            if face.size == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locations.append((startX, startY, endX, endY))

    if faces:
        faces = np.array(faces, dtype="float32")
        predictions = mask_net.predict(faces, batch_size=32)

    return (locations, predictions)

# Streamlit UI
def main():
    # Header
    st.markdown("<h1>😷 Face Mask Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-Powered Real-Time Detection System</p>", unsafe_allow_html=True)
    
    # Initialize session state
    if "detection" not in st.session_state:
        st.session_state.detection = False
    if "mask_count" not in st.session_state:
        st.session_state.mask_count = 0
    if "no_mask_count" not in st.session_state:
        st.session_state.no_mask_count = 0
    if "total_detections" not in st.session_state:
        st.session_state.total_detections = 0

    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Control buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🎥 Start Detection", key="start"):
                st.session_state.detection = True
                st.rerun()
        with btn_col2:
            if st.button("⏹️ Stop Detection", key="stop"):
                st.session_state.detection = False
                st.rerun()

    # Statistics cards
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    with stat_col2:
        st.markdown(f"""
        <div class='stats-card'>
            <div class='mask-icon'>✅</div>
            <p class='stats-number' style='color: #10b981;'>{st.session_state.mask_count}</p>
            <p class='stats-label'>With Mask</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_col1:
        st.markdown(f"""
        <div class='stats-card'>
            <div class='mask-icon'>📊</div>
            <p class='stats-number' style='color: #667eea;'>{st.session_state.total_detections}</p>
            <p class='stats-label'>Total Detections</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_col3:
        st.markdown(f"""
        <div class='stats-card'>
            <div class='mask-icon'>⚠️</div>
            <p class='stats-number' style='color: #ef4444;'>{st.session_state.no_mask_count}</p>
            <p class='stats-label'>Without Mask</p>
        </div>
        """, unsafe_allow_html=True)

    # Video feed container
    st.markdown("<div class='video-container'>", unsafe_allow_html=True)
    stframe = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    # Detection logic
    if st.session_state.detection:
        face_net, mask_net = load_models()
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Unable to access camera. Please check your camera permissions.")
            st.session_state.detection = False
            return

        while st.session_state.detection:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture frame from camera.")
                break

            frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
            locations, predictions = detect_and_predict_mask(frame, face_net, mask_net)

            current_mask = 0
            current_no_mask = 0

            for (box, pred) in zip(locations, predictions):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                
                if label == "Mask":
                    current_mask += 1
                else:
                    current_no_mask += 1
                
                confidence = max(mask, withoutMask) * 100
                label_text = "{}: {:.2f}%".format(label, confidence)

                # Draw rounded rectangle effect
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
                
                # Label background
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (startX, startY - 35), (startX + label_size[0] + 10, startY), color, -1)
                
                # Label text
                cv2.putText(frame, label_text, (startX + 5, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Update stats
            if current_mask > 0 or current_no_mask > 0:
                st.session_state.mask_count = current_mask
                st.session_state.no_mask_count = current_no_mask
                st.session_state.total_detections = current_mask + current_no_mask

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        cap.release()
    else:
        # Display placeholder when not detecting
        placeholder_html = """
        <div style='text-align: center; padding: 4rem 2rem; background: #f8f9fa; border-radius: 15px;'>
            <div style='font-size: 5rem; margin-bottom: 1rem;'>📷</div>
            <h3 style='color: #666; margin-bottom: 1rem;'>Camera Ready</h3>
            <p style='color: #999;'>Click "Start Detection" to begin</p>
        </div>
        """
        stframe.markdown(placeholder_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()