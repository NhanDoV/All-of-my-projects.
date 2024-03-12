import cv2
import numpy as np
import skimage.transform as st

def transformed(image_name, type = 'sine_transform'):

    image = cv2.imread(image_name)
    rows, cols = image.shape[0], image.shape[1]
    src_cols = np.linspace(0, cols, 20)
    src_rows = np.linspace(0, rows, 10)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]
    if type == 'sine_transform':
        dst_rows = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * 50
    elif type == 'cosine_transform':
        dst_rows = src[:, 1] - np.cos(np.linspace(0, 3 * np.pi, src.shape[0])) * 50
    elif type == 'tan_transform':
        dst_rows = src[:, 1] - np.tan(np.linspace(0, 3 * np.pi, src.shape[0])) * 50
    elif type == 'tanh_transform':
        dst_rows = src[:, 1] - np.tanh(np.linspace(0, 3 * np.pi, src.shape[0])) * 50
    elif type == 'sinh_transform':
        dst_rows = src[:, 1] - np.sinh(np.linspace(0, 3 * np.pi, src.shape[0])) * 50
    elif type == 'cosh_transform':
        dst_rows = src[:, 1] - np.cosh(np.linspace(0, 3 * np.pi, src.shape[0])) * 50
    else: 
        dst_rows = dst_rows

    dst_cols = src[:, 0]
    dst_rows *= 1.5
    dst_rows -= 1.5 * 50
    dst = np.vstack([dst_cols, dst_rows]).T
    ## piecewise-transform
    tform = st.PiecewiseAffineTransform()
    tform.estimate(src, dst)

    ## warp
    out_rows = image.shape[0] - 1.5 * 50
    out_cols = cols
    output_arr = st.warp(image, tform, output_shape=(out_rows, out_cols))

    return output_arr