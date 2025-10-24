"""
yolov8_model.py

    이 파일은 YOLOv8 세그멘테이션 모델을 로드하고,
    FastAPI의 main.py에서 이미지를 전달받아 모델 추론을 수행한 뒤,
    객체별 '감지 개수'와 '화면 내 실제 면적 비율(%)'을 계산해 반환합니다.

- 모델 로드는 서버 시작 시 단 1회만 수행됩니다.
- 추론은 run_inference(image_path) 함수로 실행됩니다.
"""

from ultralytics import YOLO
import cv2, base64, os, numpy as np

# 🎨 클래스별 색상 정의 (BGR)
CLASS_COLORS = {
    0: (0, 255, 0),     # 나무 → 초록
    1: (0, 255, 255),   # 플라스틱 → 노랑
    2: (0, 0, 255)      # 비닐 → 빨강
}

ALPHA = 0.5  # 마스크 투명도 (0~1)


class YOLOWrapper:
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)
        print(f"[YOLO] 모델 로드 완료: {weight_path}")

    def predict(self, image_path):
        # YOLO 추론
        result = self.model.predict(
            source=image_path, conf=0.8, show=False, show_boxes=False, save=True
        )[0]

        names = self.model.names
        img_h, img_w = result.orig_shape
        total_area = img_h * img_w

        ratios = {"plastic": 0.0, "vinyl": 0.0, "wood": 0.0}
        count = 0

        if result.masks is not None:
            for mask, cls_id in zip(result.masks.data, result.boxes.cls):
                mask_np = mask.cpu().numpy()

                object_area = np.sum(mask_np > 0.5)
                ratio = (object_area / total_area) * 100
                class_name = names[int(cls_id)]

                if class_name in ratios:
                    ratios[class_name] = round(ratios[class_name] + ratio, 2)
                count += 1

        # YOLO 결과 이미지 저장
        os.makedirs("temp", exist_ok=True)
        result_image_path = f"temp/result_{os.path.basename(image_path)}"
        result.save(filename=result_image_path)

        with open(result_image_path, "rb") as img_file:
            detected_image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        # 원본 이미지 Base64로 변환
        with open(image_path, "rb") as img_file:
            orig_img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        print("[비닐 확인]")
        print(ratios["vinyl"])

        return {
            "orig_img": orig_img_base64,          # 원본 이미지
            "plastic": ratios["plastic"],
            "vinyl": ratios["vinyl"],
            "wood": ratios["wood"],
            "count": count,
            "rcnn_result": detected_image_base64  # YOLO 결과 이미지 (DTO의 rcnn_result에 매핑)
        }