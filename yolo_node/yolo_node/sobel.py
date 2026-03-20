import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2

### Hough Circles used when detections found. For low granularity boxes, fall back to image's true center.
def sobel_centroids(img_arr: np.ndarray, img_pos: np.ndarray):
    def rgb_to_gray(arr):
        return np.dot(arr[..., :3], [0.299, 0.587, 0.114])
    
    def resize_dims(arr):
        h, w = arr.shape[:2]

        scale = 50 / min(h, w)
        return scale, (int(w * scale), int(h * scale))

    orig_dims = img_arr.shape[:2]

    s, dims = resize_dims(img_arr)
    arr = cv2.resize(img_arr, dims, interpolation=cv2.INTER_LINEAR)

    img = np.array(rgb_to_gray(arr), dtype=np.float64)

    img = ndimage.gaussian_filter(img, sigma=1)
    dx, dy = ndimage.sobel(img, axis=0), ndimage.sobel(img, axis=1)
    sobel_img = np.hypot(dx, dy)
    sobel_img /= np.max(sobel_img)

    thresh = np.mean(sobel_img) + 1.5 * np.std(sobel_img)
    bin_sobel_img = (sobel_img > thresh).astype(np.uint8) * 255

    ### SOBEL FILTER.
    sobel_filter = cv2.resize(bin_sobel_img, (orig_dims[1], orig_dims[0]), interpolation=cv2.INTER_NEAREST)

    # Keep only the detection closest to the image's true center.
    center = [bin_sobel_img.shape[1] / 2, bin_sobel_img.shape[0] / 2]

    circles = cv2.HoughCircles(bin_sobel_img, cv2.HOUGH_GRADIENT, dp=1.2, 
                minDist=(bin_sobel_img.shape[0]+bin_sobel_img.shape[1])/8, 
                minRadius=int(min(bin_sobel_img.shape[0], bin_sobel_img.shape[1])/3), 
                maxRadius=0, param1=1, param2=20)

    if circles is not None:
        circles = np.round(circles[0, :]).astype('int')
        x, y, _ = min(circles, key=lambda z: (z[0] - center[0]) ** 2 + (z[1] - center[1]) ** 2)

    else: # FALLBACK: Compute at image's true center.
        x, y = center

    ### CENTROID
    centroid = np.array([x, y], dtype=np.uint8) / s + img_pos[[0, 1]]
    
    return np.array(sobel_filter, dtype=np.float64), centroid
