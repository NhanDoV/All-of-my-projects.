# helper.py
# Helper functions for video frame processing and utilities

import cv2

def resize_frame(frame, scale_percent):
    """Resize a frame by a given percentage scale."""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized