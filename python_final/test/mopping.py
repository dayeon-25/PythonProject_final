import cv2
import numpy as np

# 이미지 파일 경로 설정 (두 이미지는 같은 크기여야 합니다)
destination = "C:/Users/301/Desktop/create_image/image_yolo27.png"
source = "C:/Users/301/Desktop/create_image/image_yolo18.png"

# 이미지 읽기
image_source = cv2.imread(source)
image_destination = cv2.imread(destination)

# 두 이미지를 동일한 크기로 리사이즈 (필요시)
height, width, _ = image_source.shape
image_destination = cv2.resize(image_destination, (width, height))

# 100단계에 걸쳐 모핑 진행
for i in range(101):
    percentage = i / 100.0
    # 두 이미지를 비율에 맞게 혼합
    morphed_image = cv2.addWeighted(image_source, 1 - percentage, image_destination, percentage, 0)

    # 결과 이미지 보여주기
    cv2.imshow("Morphed Image", morphed_image)
    cv2.waitKey(10)  # 100ms마다 프레임 전환

# 창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()