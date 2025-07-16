# Copyright (c) 2025 ITS
#
# All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying, use,
# modification, or distribution of this software is strictly prohibited unless
# explicitly authorized by ITS.

import os
from datetime import datetime
import shutil
import sys

def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


def get_width_height(patch_info):
    w_input = int(patch_info.split('x')[-1])
    h_input = int(patch_info.split('x')[0].split('_')[-1])
    return w_input, h_input


def parse_model_name(model_name):
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]

    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    return int(h_input), int(w_input), model_type, scale


def make_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def ensure_vggface_model():
    """
    Copy VGGFace weights to DeepFace expected path if not already there.
    """
    # Determine the source path of your local model
    if getattr(sys, 'frozen', False):  # if compiled with PyInstaller
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    src_model_path = os.path.abspath(os.path.join(base_dir, "..", "models", "face", "verify", "weights", "vgg_face_weights.h5"))
    target_dir = os.path.join(os.path.expanduser("~"), ".deepface", "weights")
    target_model_path = os.path.join(target_dir, "vgg_face_weights.h5")

    # Only copy if missing
    if not os.path.exists(target_model_path):
        os.makedirs(target_dir, exist_ok=True)
        shutil.copyfile(src_model_path, target_model_path)




