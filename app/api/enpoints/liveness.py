# Copyright (c) 2025 ITS
#
# All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying, use,
# modification, or distribution of this software is strictly prohibited unless
# explicitly authorized by ITS.

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

from app.services import liveness_service
from app.utils import image_utils

# Assuming your existing helper functions are importable
# from yourproject import img_helper, lv_check, convert_numpy_types

app = FastAPI()

class LivenessRequest(BaseModel):
    image: str

@app.post("/check_liveness")
async def checking_liveness(request: LivenessRequest):
    """Endpoint to verify liveness from a Base64 image."""
    try:
        base64_image = request.image

        if not base64_image:
            raise HTTPException(status_code=400, detail="No image provided")

        logging.info("Received request for liveness verification")

        image = image_utils.base64_to_image(base64_image)

        # Perform liveness check and return multiple metrics
        is_real, variance, movement, eye_aspect_ratio = liveness_service.is_real_person(image, return_all_metrics=True)

        # Normalize values between 0.0 and 1.0
        movement = max(0.0, min(1.0, movement))
        eye_aspect_ratio = max(0.0, min(1.0, eye_aspect_ratio))
        confidence = min(1.0, variance / 100)

        # Format as strings for consistent response
        movement_str = f"{movement:.2f}"
        eye_aspect_ratio_str = f"{eye_aspect_ratio:.2f}"
        confidence_str = f"{confidence:.2f}"

        # Create list of failure reasons, if applicable
        reasons = []
        if not is_real:
            reasons.append(f"Possible spoofing detected (low texture). Variance: {variance:.2f}")
        if movement < 0.2:
            reasons.append("Insufficient facial movement detected.")
        if eye_aspect_ratio < 0.2:
            reasons.append("Eyes appear closed or unnatural.")

        response = {
            "status": "success" if is_real else "error",
            "liveness": is_real,
            "confidence": confidence_str,
            "metrics": {
                "variance": f"{variance:.2f}",
                "movement": movement_str,
                "eyeAspectRatio": eye_aspect_ratio_str
            },
            "reasons": reasons,
            "message": "Liveness check completed successfully" if is_real else "Liveness check failed"
        }

        logging.info(f"Liveness check result: {response}")
        return JSONResponse(content={key: convert_numpy_types(value) for key, value in response.items()})

    except Exception as e:
        logging.error(f"Error processing liveness check: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing liveness check")

