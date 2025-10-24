"""
yolov8_model.py

📌 이 파일은 YOLOv8 세그멘테이션 모델을 로드하고,
   FastAPI의 main.py에서 이미지를 전달받아 모델 추론을 수행한 뒤,
   객체별 '감지 개수'와 '화면 내 실제 면적 비율(%)'을 계산해 반환합니다.

- 모델 로드는 서버 시작 시 단 1회만 수행됩니다.
- 추론은 run_inference(image_path) 함수로 실행됩니다.
"""

from ultralytics import YOLO
import cv2, base64, os
import numpy as np


# ==============================================
# ✅ 1. YOLO 모델을 전역에서 한 번만 로드
# ==============================================
# 서버 시작 시 한 번만 모델을 메모리에 올려두면
# 매 요청마다 모델을 새로 불러오지 않아 속도가 크게 향상됩니다.
model = YOLO("C:/Users/301/Desktop/final_result80/runs/yolo_seg_train/weights/best.pt")



# ==============================================
# ✅ 2. YOLO 추론 및 결과 처리 함수
# ==============================================
def run_inference(image_path: str):
    """
    ✅ main.py에서 DTO에 맞게 조합할 수 있도록
       YOLO 분석 결과만 반환하는 함수로 변경
    Returns
    -------
    dict : {
        "orig_img": base64 인코딩된 이미지,
        "plastic": float,
        "vinyl": float,
        "wood": float,
        "count": int
    }
    """
    print("[LOG] YOLO 추론 시작...")
    results = model.predict(
        source=image_path,
        conf=0.25,
        show=False,
        boxes=False,
        save=True
    )

    result = results[0]
    names = model.names
    counts = {}
    ratios = {}
    total_detected = 0

    img_h, img_w = result.orig_shape
    total_area = img_h * img_w

    if result.masks is not None:
        for mask, cls_id in zip(result.masks.data, result.boxes.cls):
            mask_np = mask.cpu().numpy()
            object_area = np.sum(mask_np > 0.5)
            ratio = (object_area / total_area) * 100

            class_name = names[int(cls_id)]
            counts[class_name] = counts.get(class_name, 0) + 1
            ratios[class_name] = round(ratios.get(class_name, 0) + ratio, 2)
            total_detected += 1

    os.makedirs("temp", exist_ok=True)
    result_path = f"temp/result_{os.path.basename(image_path)}"
    result.save(filename=result_path)

    with open(result_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

    # ✅ DTO에 필요한 필드명으로 맞춰서 반환
    return {
        "orig_img": encoded_image,
        "plastic": ratios.get("plastic", 0.0),
        "vinyl": ratios.get("vinyl", 0.0),
        "wood": ratios.get("wood", 0.0),
        "count": total_detected
    }