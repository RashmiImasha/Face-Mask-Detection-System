import streamlit as st
import cv2

from detection_utils import (
    detect_and_predict_mask,
    draw_detection_results,
    resize_frame,
)
from alert_system import AlertSystem
from model_loader import load_models


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
    .alert-box {
        background: rgba(239, 68, 68, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #dc2626;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
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
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all session state variables"""
    if "detection" not in st.session_state:
        st.session_state.detection = False
    if "mask_count" not in st.session_state:
        st.session_state.mask_count = 0
    if "no_mask_count" not in st.session_state:
        st.session_state.no_mask_count = 0
    if "total_detections" not in st.session_state:
        st.session_state.total_detections = 0
    if "total_violations" not in st.session_state:
        st.session_state.total_violations = 0
    if "current_alert" not in st.session_state:
        st.session_state.current_alert = None

# Load model
@st.cache_resource
def load_detection_models():
    return load_models()

def get_alert_system():
    if "alert_system" not in st.session_state:
        st.session_state.alert_system = AlertSystem(alert_cooldown=5)
    return st.session_state.alert_system


def render_header():
    """Render application header"""
    st.markdown("<h1>😷 Face Mask Detection System with Alerts</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-Powered Detection with Beep Sound & Voice Warnings</p>", unsafe_allow_html=True)


def render_statistics():
    """Render statistics cards"""
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    with stat_col1:
        st.markdown(f"""
        <div class='stats-card'>
            <div style='font-size: 2.5rem;'>📊</div>
            <p class='stats-number' style='color: #667eea;'>{st.session_state.total_detections}</p>
            <p class='stats-label'>Total Detections</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_col2:
        st.markdown(f"""
        <div class='stats-card'>
            <div style='font-size: 2.5rem;'>✅</div>
            <p class='stats-number' style='color: #10b981;'>{st.session_state.mask_count}</p>
            <p class='stats-label'>With Mask</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_col3:
        st.markdown(f"""
        <div class='stats-card'>
            <div style='font-size: 2.5rem;'>⚠️</div>
            <p class='stats-number' style='color: #ef4444;'>{st.session_state.no_mask_count}</p>
            <p class='stats-label'>Without Mask</p>
        </div>
        """, unsafe_allow_html=True)
    
    

def render_control_buttons():
    """Render start/stop buttons"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            if st.button("🎥 Start Detection", key="start"):
                st.session_state.detection = True
                st.rerun()
                
        
        with btn_col2:
            if st.button("⏹️ Stop Detection", key="stop"):
                st.session_state.detection = False
                st.session_state.current_alert = None
                st.rerun()
               
def render_alert_notification():
    if st.session_state.current_alert:
        st.markdown(f"🚨 ALERT: {st.session_state.current_alert}")


# ============================================
# MAIN DETECTION LOOP
# ============================================
def run_detection(face_net, mask_net, alert_system, stframe):
    """Run the main detection loop"""
    # Ensure audio is ready (in case of restart)
    alert_system._init_audio()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Unable to access camera. Please check your camera permissions.")
        st.session_state.detection = False
        return

    try:
        while st.session_state.detection:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture frame from camera.")
                break

            frame = resize_frame(frame, width=640)
            locations, predictions = detect_and_predict_mask(frame, face_net, mask_net)
            frame, mask_count, no_mask_count = draw_detection_results(frame, locations, predictions, show_confidence=True)

            # Update statistics
            st.session_state.mask_count = mask_count
            st.session_state.no_mask_count = no_mask_count
            st.session_state.total_detections = mask_count + no_mask_count

            # Trigger alert
            if no_mask_count > 0:
                alert_triggered = alert_system.trigger_alert(no_mask_count)
                if alert_triggered:
                    st.session_state.current_alert = (
                        "1 person without mask detected!"
                        if no_mask_count == 1
                        else f"{no_mask_count} persons without masks detected!"
                    )
            else:
                st.session_state.current_alert = None

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    finally:
        cap.release()
      

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Load models and alert system
    face_net, mask_net = load_detection_models()
    alert_system = get_alert_system()
    
    # Render UI
    render_header()
    render_control_buttons()
    render_statistics()
    render_alert_notification()
    
    # Video feed container
    st.markdown("<div class='video-container'>", unsafe_allow_html=True)
    stframe = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    
    
    
    # Run detection or show placeholder
    if st.session_state.detection:
        run_detection(face_net, mask_net, alert_system, stframe)
    else:
        # Display placeholder when not detecting
        placeholder_html = """
        <div style='text-align: center; padding: 4rem 2rem; background: #f8f9fa; border-radius: 15px;'>
            <div style='font-size: 5rem; margin-bottom: 1rem;'>📷</div>
            <h3 style='color: #666; margin-bottom: 1rem;'>Camera Ready</h3>
            <p style='color: #999;'>Click "Start Detection" to begin monitoring</p>
            <p style='color: #999; margin-top: 1rem;'>
                When a person without mask is detected:<br>
                🔔 Beep sound will play<br>
                🔊 Voice warning will announce the person count
            </p>
        </div>
        """
        stframe.markdown(placeholder_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
