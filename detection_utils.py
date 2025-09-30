# """
# Face detection and mask prediction utilities
# """
# import cv2
# import numpy as np
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array


# # Face detection and mask prediction
# def detect_and_predict_mask(frame, face_net, mask_net):
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

#     face_net.setInput(blob)
#     detections = face_net.forward()
    
#     faces = []
#     locations = []
#     predictions = []

#     for i in range(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]

#         if confidence > 0.5:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")

#             (startX, startY) = (max(0, startX), max(0, startY))
#             (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

#             face = frame[startY:endY, startX:endX]

#             if face.size == 0:
#                 continue
#             face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#             face = cv2.resize(face, (224, 224))
#             face = img_to_array(face)
#             face = preprocess_input(face)

#             faces.append(face)
#             locations.append((startX, startY, endX, endY))

#     if faces:
#         faces = np.array(faces, dtype="float32")
#         predictions = mask_net.predict(faces, batch_size=32)

#     return (locations, predictions)

"""
detection_utils.py
Face detection and mask prediction utilities
"""

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


def detect_and_predict_mask(frame, face_net, mask_net, confidence_threshold=0.5):
    """
    Detect faces in a frame and predict mask status
    
    Args:
        frame: Input image frame
        face_net: Face detection model
        mask_net: Mask classification model
        confidence_threshold: Minimum confidence for face detection (default: 0.5)
    
    Returns:
        tuple: (locations, predictions) where locations are face bounding boxes
               and predictions are mask classification results
    """
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()
    
    faces = []
    locations = []
    predictions = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure bounding box is within frame boundaries
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract face region
            face = frame[startY:endY, startX:endX]

            if face.size == 0:
                continue
            
            # Preprocess face for mask detection
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locations.append((startX, startY, endX, endY))

    # Make mask predictions if faces detected
    if faces:
        faces = np.array(faces, dtype="float32")
        predictions = mask_net.predict(faces, batch_size=32)

    return (locations, predictions)


def draw_detection_results(frame, locations, predictions, show_confidence=True):
    """
    Draw bounding boxes and labels on frame
    
    Args:
        frame: Input image frame
        locations: List of face bounding boxes
        predictions: List of mask predictions
        show_confidence: Whether to display confidence scores
    
    Returns:
        tuple: (annotated_frame, mask_count, no_mask_count)
    """
    mask_count = 0
    no_mask_count = 0
    
    for (box, pred) in zip(locations, predictions):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        
        # Determine label and color
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        # Update counts
        if label == "Mask":
            mask_count += 1
        else:
            no_mask_count += 1
        
        # Calculate confidence
        confidence = max(mask, withoutMask) * 100
        
        # Create label text
        if show_confidence:
            label_text = "{}: {:.2f}%".format(label, confidence)
        else:
            label_text = label

        # Draw bounding box
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
        
        # Draw label background
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (startX, startY - 35), 
                     (startX + label_size[0] + 10, startY), color, -1)
        
        # Draw label text
        cv2.putText(frame, label_text, (startX + 5, startY - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame, mask_count, no_mask_count


def resize_frame(frame, width=640):
    """
    Resize frame maintaining aspect ratio
    
    Args:
        frame: Input image frame
        width: Target width in pixels
    
    Returns:
        Resized frame
    """
    height = int(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, height))


def validate_camera(camera_index=0):
    """
    Validate if camera is accessible
    
    Args:
        camera_index: Camera index (default: 0 for primary camera)
    
    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        return False, "Unable to access camera. Please check permissions."
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return False, "Camera opened but unable to capture frame."
    
    return True, None