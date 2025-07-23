import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import tempfile

# load model
@st.cache_resource
def load_models():
    face_net = cv2.dnn.readNet(
        "face_detector/deploy.prototxt",
        "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    )
    mask_net = load_model("mask_detector.model.h5")
    return face_net, mask_net

# face detection and mask prediction
def detect_and_predict_mask(frame, face_net, mask_net):
    # construct a blob using dimensions of the frame
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    face_net.setInput(blob)
    detections = face_net.forward()
    
    # initialize lists for faces, their locations, and mask predictions
    faces = []
    locations = []
    predictions = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.5:
            # compute the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the frame dimensions
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract, convert, resize, and preprocess the face ROI
            face = frame[startY:endY, startX:endX]

            if face.size == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # append the face and its bounding box
            faces.append(face)
            locations.append((startX, startY, endX, endY))

    # making predictions
    if faces :
        faces = np.array(faces, dtype="float32")
        predictions = mask_net.predict(faces, batch_size=32)

    return (locations, predictions)

# streamlit UI
def main():
    st.title("Face Mask Detection")
    st.markdown("""
        Detect whether a person is wearing a face mask using your webcam.
        \nClick the button below to start detecting.
    """)

    # Placeholder for the video feed
    stframe = st.empty()

    # Session state to keep track of detection toggle
    if "detection" not in st.session_state:
        st.session_state.detection = False

    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Detection"):
            st.session_state.detection = True
    with col2:
        if st.button("Stop Detection"):
            st.session_state.detection = False
            st.write("Detection stopped.")
            stframe.empty()

    if st.session_state.detection:
        face_net, mask_net = load_models()
        cap = cv2.VideoCapture(0)

        while st.session_state.detection:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (400, int(frame.shape[0] * 400 / frame.shape[1])))
            locations, predictions = detect_and_predict_mask(frame, face_net, mask_net)

            for (box, pred) in zip(locations, predictions):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()


if __name__ == "__main__":
    main()
    