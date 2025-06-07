import base64
import io
import tempfile
import traceback
from typing import Any

import cv2
import numpy as np
from PIL import Image
from numpy import ndarray

from app.middleware.logging import logger


def base64_to_image(base64_str: str) -> ndarray | None | Any:
    """Convert base64 string to OpenCV image."""
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        logger.debug(f"Image converted successfully: shape={cv_image.shape}")
        print(f"Image converted successfully: shape={cv_image.shape}")
        return cv_image
    except Exception as e:
        logger.error(f"Error converting base64 to image: {str(e)}", exc_info=True)
        print(f"Error converting base64 to image: {str(e)}")
        return None


def save_temp_image(image) -> str | dict:
    """
    Saves an image to a temporary JPEG file and returns the file path.

    If the input is a NumPy array, it will be converted to a PIL Image.
    Returns the file path as a string, or a dict containing error information if an exception occurs.
    """
    try:
        logger.info("Saving received image temporarily.")

        # Convert NumPy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Create a temporary file with a .jpg suffix
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        image.save(temp_file, format="JPEG")
        temp_file.close()  # Close the file to flush and release for external access

        logger.info("Temporary image file created successfully.")
        logger.debug(f"Temporary file path: {temp_file.name}")

        return temp_file.name

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error("Failed to create temporary image file:\n" + error_trace)
        return {"error": str(e), "stack_trace": error_trace}
