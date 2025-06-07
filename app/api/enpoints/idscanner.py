# Copyright (c) 2025 ITS
#
# All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying, use,
# modification, or distribution of this software is strictly prohibited unless
# explicitly authorized by ITS.

import base64
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import cv2
import numpy as np
from fastapi import File, UploadFile, HTTPException, APIRouter
from paddleocr import PaddleOCR
from pydantic import BaseModel
from ultralytics import YOLO

from app.middleware.logging import logger

router = APIRouter()


# Function to get the correct path based on environment (normal or packaged)
def get_model_path(relative_path):
    import sys
    import os
    if getattr(sys, 'frozen', False):  # If running as a packaged executable
        base_path = sys._MEIPASS  # Path where the executable stores its files
    else:  # If running in development environment
        base_path = os.path.dirname(__file__)  # Current directory
    return os.path.join(base_path, relative_path)


# Paths to the local models
det_model_dir = get_model_path('../../models/det/en_PP-OCRv3_det_infer')
rec_model_dir = get_model_path('../../models/rec/latin_PP-OCRv3_rec_infer')
cls_model_dir = get_model_path('../../models/cls/ch_ppocr_mobile_v2.0_cls_infer')


# State tracking
class ExtractionState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.extracted_data = {}
        self.last_detected = {}
        self.extraction_complete = False
        self.last_photo_saved = 0
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.extraction_start_time = time.time()
        self.photo_base64 = None  # Add photo_base64 field

    def update_field(self, field, value, confidence):
        # Update if confidence is higher or field not yet detected
        # Also update if the field was detected a while ago (time decay)
        current_time = time.time()
        time_threshold = 5.0  # seconds

        should_update = (
                field not in self.extracted_data or
                confidence > self.extracted_data[field]['confidence'] or
                (field in self.last_detected and
                 current_time - self.last_detected[field] > time_threshold)
        )

        if should_update:
            self.extracted_data[field] = {
                'value': value,
                'confidence': confidence,
                'timestamp': current_time
            }
            self.last_detected[field] = current_time

    def check_completion(self, required_fields, confidence_threshold):
        """Check if all required fields have been extracted with sufficient confidence"""
        for field in required_fields:
            if (field not in self.extracted_data or
                    self.extracted_data[field]['confidence'] < confidence_threshold):
                return False
        # Additionally check if we have the photo_base64 for the photo field
        if 'photo' in required_fields and not self.photo_base64:
            return False
        return True

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "extracted_data": self.extracted_data,
            "extraction_complete": self.extraction_complete,
            "extraction_duration": time.time() - self.extraction_start_time,
            "photo_base64": self.photo_base64
        }

    def save_state(self, directory="extraction_states"):
        """Save current state to a JSON file"""
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = os.path.join(directory, f"extraction_{self.session_id}.json")
        with open(filename, 'w') as f:
            # Convert to a serializable format
            data = {
                "session_id": self.session_id,
                "extracted_data": {
                    k: {
                        "value": v["value"],
                        "confidence": v["confidence"],
                        "timestamp": v["timestamp"]
                    } for k, v in self.extracted_data.items()
                },
                "extraction_complete": self.extraction_complete,
                "extraction_duration": time.time() - self.extraction_start_time,
                "photo_base64": self.photo_base64,  # Include photo_base64 in saved state
                "timestamp": datetime.now().isoformat()
            }
            json.dump(data, f, indent=2)

        return filename


# Global variables
state = ExtractionState()
REQUIRED_FIELDS = ["id_number", "name", "photo"]
CONFIDENCE_THRESHOLD = 0.6
PHOTO_SAVE_INTERVAL = 2  # seconds


# Pydantic models for request and response
class ExtractedField(BaseModel):
    value: str
    confidence: float
    timestamp: Optional[float] = None


class ProcessFrameResponse(BaseModel):
    session_id: str
    extracted_data: Dict[str, ExtractedField]
    extraction_complete: bool
    missing_fields: List[str]
    photo_base64: Optional[str] = None


class ResetExtractionRequest(BaseModel):
    session_id: Optional[str] = None


# Load models at startup
@router.on_event("startup")
async def startup_event():
    global ocr, model

    # Create directories
    os.makedirs("extraction_photos", exist_ok=True)
    os.makedirs("extraction_states", exist_ok=True)

    # Load PaddleOCR model
    logger.info("Loading OCR model...")
    try:
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang="pt",
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            cls_model_dir=cls_model_dir
        )
        logger.info("OCR model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading OCR model: {str(e)}")
        raise

    # Load YOLO model
    logger.info("Loading YOLO model...")
    try:
        current_file = Path(__file__).resolve()
        app_dir = current_file.parent.parent.parent
        model_path = app_dir / "models" / "det" / "best_recent.pt"
        model = YOLO(str(model_path))
        logger.info(f"YOLO model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Error loading YOLO model: {str(e)}")
        raise


@router.post("/process-frame/", response_model=ProcessFrameResponse)
async def process_frame(file: UploadFile = File(...)):
    global state

    # Read and decode the image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    current_time = time.time()
    photo_base64 = None

    # Run YOLO detection on the frame
    results = model(frame)

    # Process each detection
    for result in results:
        for box in result.boxes:
            # Get the coordinates as integers
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class index
            label = model.names[cls]  # Get label name

            # Confidence threshold
            if conf < 0.3:
                continue

            # Crop detected region
            cropped_region = frame[y1:y2, x1:x2]

            if cropped_region.size == 0:
                continue  # Skip empty regions

            # If this is a photo and enough time has passed since last save
            if label == "photo" and (current_time - state.last_photo_saved) > PHOTO_SAVE_INTERVAL:
                # Save photo to file
                photo_path = os.path.join("extraction_photos", f"{state.session_id}_photo.jpg")
                cv2.imwrite(photo_path, cropped_region)
                logger.info(f"Photo saved as {photo_path}")
                state.last_photo_saved = current_time

                # Convert to base64 for API response
                _, buffer = cv2.imencode('.jpg', cropped_region)
                photo_base64 = base64.b64encode(buffer).decode('utf-8')
                logger.info("Photo converted to base64 successfully")

                # Update state with both path and base64
                state.update_field("photo", photo_path, conf)
                state.photo_base64 = photo_base64  # Store base64 in state

            # For other fields, perform OCR
            elif label in ["id_number", "name"]:
                # Convert to grayscale for better OCR
                gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)

                # Apply OCR using PaddleOCR
                result = ocr.ocr(gray, cls=True)

                # Extract text
                if result and result[0]:
                    # Get the text with the highest confidence
                    best_text = ""
                    best_conf = 0
                    for line in result[0]:
                        text = line[1][0]
                        ocr_conf = float(line[1][1])
                        if ocr_conf > best_conf:
                            best_text = text
                            best_conf = ocr_conf

                    if best_text and best_conf > 0.4:  # Lowered from 0.6
                        # Calculate combined confidence score
                        combined_conf = (conf + best_conf) / 2

                        # Special processing for ID number
                        if label == "id_number":
                            # Look for pattern Nº: followed by numbers and ending with a letter
                            match = re.search(r'N[º°:]\s*:?\s*(\d+[A-Za-z])', best_text, re.IGNORECASE)
                            if match:
                                id_number = match.group(1)
                                # Remove any leading 'N' if it was accidentally included
                                if id_number.startswith('N'):
                                    id_number = id_number[1:]

                                # If the last character is a digit, try to convert it to a letter
                                if id_number[-1].isdigit():
                                    # Common OCR confusions for letters
                                    digit_to_letter = {
                                        '0': 'Q',
                                        '1': 'I',
                                        '8': 'B',
                                        '5': 'S'
                                    }
                                    last_char = id_number[-1]
                                    if last_char in digit_to_letter:
                                        id_number = id_number[:-1] + digit_to_letter[last_char]
                                best_text = id_number
                            else:
                                # If no match found, try to extract numbers followed by a letter
                                clean_text = re.sub(r'[^0-9A-Z]', '', best_text.upper())
                                if clean_text:
                                    # Remove any leading 'N' if it was accidentally included
                                    if clean_text.startswith('N'):
                                        clean_text = clean_text[1:]

                                    # Ensure the last character is a letter
                                    if clean_text[-1].isdigit():
                                        # Common OCR confusions for letters
                                        digit_to_letter = {
                                            '0': 'O',
                                            '1': 'I',
                                            '8': 'B',
                                            '5': 'S'
                                        }
                                        last_char = clean_text[-1]
                                        if last_char in digit_to_letter:
                                            clean_text = clean_text[:-1] + digit_to_letter[last_char]
                                best_text = clean_text

                            # Log the ID number processing
                            logger.info(f"Processed ID number: {best_text}")

                        # Update state
                        state.update_field(label, best_text, combined_conf)
                        logger.info(f"Detected {label}: {best_text} (conf: {combined_conf:.2f})")

    # Check if extraction is complete
    state.extraction_complete = state.check_completion(REQUIRED_FIELDS, CONFIDENCE_THRESHOLD)

    # If extraction is complete, save the state
    if state.extraction_complete:
        state_file = state.save_state()
        logger.info(f"Extraction complete! State saved to {state_file}")

    # Prepare response data, ensuring photo_base64 is included if available
    response_data = {
        "session_id": state.session_id,
        "extracted_data": {
            k: ExtractedField(
                value=v["value"],
                confidence=v["confidence"],
                timestamp=v["timestamp"]
            ) for k, v in state.extracted_data.items()
        },
        "extraction_complete": state.extraction_complete,
        "missing_fields": [field for field in REQUIRED_FIELDS
                           if field not in state.extracted_data or
                           state.extracted_data[field]['confidence'] < CONFIDENCE_THRESHOLD],
        "photo_base64": getattr(state, 'photo_base64', None)  # Get photo_base64 from state
    }

    return ProcessFrameResponse(**response_data)


@router.post("/reset-extraction/")
async def reset_extraction(request: ResetExtractionRequest = None):
    global state
    state.reset()
    if request and request.session_id:
        state.session_id = request.session_id

    return {"message": "Extraction state reset", "session_id": state.session_id}


@router.get("/extraction-status/")
async def get_extraction_status():
    global state

    # Determine missing fields
    missing_fields = [field for field in REQUIRED_FIELDS
                      if field not in state.extracted_data or
                      state.extracted_data[field]['confidence'] < CONFIDENCE_THRESHOLD]

    return {
        "session_id": state.session_id,
        "extraction_complete": state.extraction_complete,
        "missing_fields": missing_fields,
        "extracted_fields": list(state.extracted_data.keys()),
        "elapsed_time": time.time() - state.extraction_start_time
    }
