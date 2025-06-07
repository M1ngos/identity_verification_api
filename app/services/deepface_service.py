# Copyright (c) 2025 ITS
#
# All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying, use,
# modification, or distribution of this software is strictly prohibited unless
# explicitly authorized by ITS.

import traceback
from typing import Any

import gdown
import os
import tqdm
from PIL import Image
from deepface import DeepFace

from app.middleware.logging import logger
from app.utils import image_utils


def verify_images(img1: Image.Image, img2: Image.Image) -> dict[str, str] | dict[str, Any] | None:
    """
    Compares two images using DeepFace to check for facial similarity.

    :param img1: First image (PIL Image).
    :param img2: Second image (PIL Image).
    :return: Dictionary containing match result and distance, or error details if failed.
    """
    try:
        # Disable tqdm progress bar and gdown downloads to avoid console clutter and unwanted downloads
        tqdm.tqdm.__init__ = lambda *args, **kwargs: None
        tqdm.tqdm.update = lambda *args, **kwargs: None
        tqdm.tqdm.close = lambda *args, **kwargs: None
        gdown.download = lambda *args, **kwargs: None
        os.environ["DEEPFACE_DOWNLOAD"] = "0"

        # Save both images to temporary files
        img1_path = image_utils.save_temp_image(img1)
        img2_path = image_utils.save_temp_image(img2)

        # Validate paths
        if not isinstance(img1_path, str) or not os.path.exists(img1_path) or \
                not isinstance(img2_path, str) or not os.path.exists(img2_path):
            logger.error("One or both temporary image files were not created properly.")
            return {"error": "One or both temporary image files were not created!"}

        try:
            # Run DeepFace verification
            result = DeepFace.verify(
                img1_path,
                img2_path,
                model_name="VGG-Face",
                detector_backend="opencv"
            )
        finally:
            # Clean up temporary image files
            os.remove(img1_path)
            os.remove(img2_path)

        return {
            "match": result["verified"],
            "distance": result["distance"]
        }

    except Exception as e:
        # Capture full stack trace and log the error
        error_trace = traceback.format_exc()
        logger.error("Error during image comparison:\n" + error_trace)

        return {
            "error": str(e),
            "stack_trace": error_trace
        }
