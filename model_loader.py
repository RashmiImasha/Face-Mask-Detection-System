import cv2
from tensorflow.keras.models import load_model

def load_models(prototxt_path="face_detector/deploy.prototxt",
               weights_path="face_detector/res10_300x300_ssd_iter_140000.caffemodel",
               model_path="mask_detector.model.h5"):
    """    
    Args:
        prototxt_path: Path to face detection prototxt
        weights_path: Path to face detection weights
        model_path: Path to mask detection model
    
    """
    print("[INFO] Loading face detection model...")
    face_net = cv2.dnn.readNet(prototxt_path, weights_path)
    
    print("[INFO] Loading mask detection model...")
    mask_net = load_model(model_path)
    
    print("[INFO] Models loaded successfully!")
    
    return face_net, mask_net
    
    

    