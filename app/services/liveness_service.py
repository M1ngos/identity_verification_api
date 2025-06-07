import cv2, os
import numpy as np
from PIL import Image
import dlib

from app.utils.image_utils import convert_image_to_mat

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("app/models/face/shape_predictor_68_face_landmarks.dat")  # Required to detect eyes


def is_real_person(img: Image, return_all_metrics=False) -> bool:
    # Convert Image (PIL) to OpenCV Mat (NumPy array)
    image = convert_image_to_mat(img)

    if image is None or image.size == 0:
        print("Image could not be processed.")
        return (False, 0.0) if return_all_metrics else False

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian operator to detect texture
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Compute variance of the Laplacian result (used as texture quality)
    variance = laplacian.var()

    # Initialize movement and eye aspect ratio values
    movement = 0.0  # With only 1 frame, movement is 0
    eye_ratio = 0.0

    # Face detection
    faces = detector(image)
    if len(faces) > 0:
        face = faces[0]  # Use only the first detected face
        landmarks = predictor(image, face)

        # Get left and right eye landmarks (according to dlib's 68-point model)
        left_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(36, 42)])
        right_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(42, 48)])

        # Calculate eye aspect ratio
        eye_ratio = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        eye_ratio = max(0.0, min(1.0, eye_ratio))  # Clamp to [0, 1]

    # Get the base directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Relative path to the anti-spoofing model directory
    model_dir = os.path.join(base_dir, "app", "models", "face", "anti_spoof_models")

    label = test(
        image=image,
        model_dir=model_dir,
        device_id=0
    )

    print(f"SPOOFING VALUE {label}")

    # Final decision based on variance, eye ratio, and spoofing model
    is_real = variance > 14 and eye_ratio > 0.2 and label == 1

    if return_all_metrics:
        return is_real, variance, movement, eye_ratio
    return is_real


def eye_aspect_ratio(eye):
    """Computes the ratio between the vertical and horizontal eye landmarks."""
    a = np.linalg.norm(eye[1] - eye[5])
    b = np.linalg.norm(eye[2] - eye[4])
    c = np.linalg.norm(eye[0] - eye[3])
    return (a + b) / (2.0 * c)

