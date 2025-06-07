# Copyright (c) 2025 ITS
#
# All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying, use,
# modification, or distribution of this software is strictly prohibited unless
# explicitly authorized by ITS.

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.enpoints.idscanner import router as id_scanner_router
from app.api.enpoints.comparison import router as image_comparison_router

# Initialize FastAPI app
app = FastAPI(
    title="Liveness | ID Card Extraction API",
    description="API for extracting data from ID cards,"
          " checking liveness"
          " and comparing faces"
          " using webcam frames",
    version="0.1.2"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# routes/endpoints
app.include_router(id_scanner_router, prefix="/id")
app.include_router(image_comparison_router, prefix="/image")

# Events
@app.on_event("startup")
async def startup_event():
    logging.info("Starting Liveness and IDScanner Server")

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Shutting Down Liveness and IDScanner Server")