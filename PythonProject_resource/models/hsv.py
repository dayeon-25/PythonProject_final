import cv2


def pick_hsv(event, x, y, flags, param):
   if event == cv2.EVENT_LBUTTONDOWN:
       hsv_val = hsv[y, x]
       print(f"픽셀 위치 ({x}, {y}) HSV 값: {hsv_val}")


#IMAGE_PATH = "C:/Users/dayeon/Desktop/create_image/ChatGPT_Image.png"
#IMAGE_PATH = "C:/Users/301/Desktop/create_image/image_yolo31.png"
#IMAGE_PATH = "C:/Users/301/PycharmProjects/PythonProject_final/image/plastic/26_jpeg_jpg.rf.44ff45f754439799f5e1de0f7da7d00d.jpg"
IMAGE_PATH = "C:/Users/301/Desktop/create_image/image_yolo31.png"


img = cv2.imread(IMAGE_PATH)  # 위 이미지 파일로 변경
img = cv2.resize(img, (600, 600))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


cv2.imshow('image', img)
cv2.setMouseCallback('image', pick_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
