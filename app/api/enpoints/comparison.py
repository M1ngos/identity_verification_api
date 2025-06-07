# Copyright (c) 2025 ITS
#
# All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying, use,
# modification, or distribution of this software is strictly prohibited unless
# explicitly authorized by ITS.

from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from starlette.responses import JSONResponse

from app.middleware.logging import logger
from app.services import deepface_service
from app.utils import image_utils

router = APIRouter()


class ImageCompareRequest(BaseModel):
    image1: str
    image2: str


@router.post("/compare_images")
async def compare_images(data: ImageCompareRequest):
    try:
        # Validate required parameters
        if not data.image1:
            raise HTTPException(status_code=400, detail="The 'image1' parameter is required")
        if not data.image2:
            raise HTTPException(status_code=400, detail="The 'image2' parameter is required")

        logger.info("Received request for image comparison")

        # Convert base64 to image objects
        image1 = image_utils.base64_to_image(data.image1)
        image2 = image_utils.base64_to_image(data.image2)

        logger.info("Starting comparison...")

        # Call DeepFace method to compare the images
        result = deepface.verify_images(image1, image2)

        logger.info(f"Result: {result}")
        logger.info("Comparison completed!")

        # Return the result
        return JSONResponse(content=result)

    except Exception as err:
        logger.error(f"Error during comparison: {str(err)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(err)})
