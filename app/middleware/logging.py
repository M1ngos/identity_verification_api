# Copyright (c) 2025 ITS
#
# All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying, use,
# modification, or distribution of this software is strictly prohibited unless
# explicitly authorized by ITS.

import logging
import os

# Define log directory
log_dir = "C:\\captacao\\logs"
os.makedirs(log_dir, exist_ok=True)  # Ensure the log folder exists

# Create logger
logger = logging.getLogger("imageProcessorLogger")
logger.setLevel(logging.DEBUG)  # Capture all levels: DEBUG, INFO, ERROR, etc.

# Formatter for all handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Console handler (optional, for debugging in terminal)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler for INFO logs
info_handler = logging.FileHandler(os.path.join(log_dir, "imageProcessor_INFO_log.txt"))
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)
logger.addHandler(info_handler)

# File handler for ERROR logs
error_handler = logging.FileHandler(os.path.join(log_dir, "imageProcessor_ERROR_log.txt"))
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)
logger.addHandler(error_handler)
