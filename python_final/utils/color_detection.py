import cv2
import numpy as np

# pip install opencv - python
# pip install cv2

def detect_colors(img):
    """
    이미지에서 특정 색상(R,G,B) 픽셀 수 계산
    :param img: OpenCV 이미지 배열
    :return: 각 색상별 픽셀 개수 dict
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    colors = {
        "red": ((0, 50, 50), (10, 255, 255)),
        "green": ((50, 50, 50), (70, 255, 255)),
        "blue": ((100, 50, 50), (130, 255, 255))
    }
    color_counts = {}
    for color, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        color_counts[color] = int(np.sum(mask > 0))
    return color_counts