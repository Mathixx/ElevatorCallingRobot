from ultralytics import YOLO
import cv2
import numpy as np
from yolo_node.sobel import sobel_centroids
import matplotlib.pyplot as plt

def frame_boxes(preds, img):
    bboxes, sobel_maps, centroids, classes = [], [], [], []

    for box in preds.boxes:
        classes.append(box.cls)

        box_coords = box.xyxy[0].tolist()
        box_coords = np.floor(box_coords).astype(int)

        img_sub = img[box_coords[1]:box_coords[3], 
                    box_coords[0]:box_coords[2], ...]

        s, c = sobel_centroids(img_arr=img_sub, img_pos=box_coords)
        s = np.stack([s]*3, axis=-1)

        ## TODO: Plot bounding boxes & labels.
        bboxes.append(box_coords)
        sobel_maps.append(s)
        centroids.append(c)

    return bboxes, sobel_maps, centroids, classes
