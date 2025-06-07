import os

LICENSE_TEXT = """\
# Copyright (c) 2025 ITS
#
# All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying, use,
# modification, or distribution of this software is strictly prohibited unless
# explicitly authorized by ITS.

"""

def prepend_license(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    if LICENSE_TEXT.strip() in content:
        return  # Skip if license is already added
    with open(file_path, 'w') as f:
        f.write(LICENSE_TEXT + content)

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for name in files:
            if name.endswith(".py"):
                full_path = os.path.join(root, name)
                prepend_license(full_path)

# Replace with the path to your code
process_directory("C:\\YOUR\\PATH\\TO-PROJECT\\identity-verification_api\\app"
)
