import cv2
import numpy as np

from nptyping import NDArray
from typing import Any

from environment import env


def shape_predictor(image: NDArray[(Any, Any, 3), np.uint8]) -> str:
    threshold = env.threshold_binary
    epsilon = env.approx_poly_dp_epsilon

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_thresh = cv2.threshold(image_gray, threshold, 255, cv2.THRESH_BINARY)
    image_contours, _ = cv2.findContours(image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(image_contours) > 0:
        contour = image_contours[0]
    else:
        return 'shape not found'

    perimeter = cv2.arcLength(contour, True)
    contour_approx = cv2.approxPolyDP(contour, epsilon*perimeter, True)
    num_vertices = len(contour_approx)

    return {
        num_vertices < 3: 'unknown',
        num_vertices == 3: 'triangle',
        num_vertices == 4: 'rectangle',
        num_vertices == 5: 'pentagon',
        num_vertices == 6: 'hexagon',
        num_vertices > 6: 'circle',
    }[True]
