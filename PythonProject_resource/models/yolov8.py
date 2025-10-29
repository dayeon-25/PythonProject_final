"""
yolov8_model.py

    이 파일은 YOLOv8 세그멘테이션 모델을 로드하고,
    FastAPI의 main.py에서 이미지를 전달받아 모델 추론을 수행한 뒤,
    객체별 '감지 개수'와 '화면 내 실제 면적 비율(%)'을 계산해 반환합니다.

- 모델 로드는 서버 시작 시 단 1회만 수행됩니다.
- 추론은 run_inference(image_path) 함수로 실행됩니다.
"""

# ================ 검출 비율 수정하기 전 버전 ================
from ultralytics import YOLO
import cv2, base64, os, numpy as np

# 🎨 클래스별 색상 정의 (BGR)
CLASS_COLORS = {
    0: (255, 0, 0),     # 나무 → 초록
    1: (0, 255, 255),   # 플라스틱 → 노랑
    2: (0, 0, 255)      # 비닐 → 빨강
}

ALPHA = 0.3  # 마스크 투명도


class YOLOWrapper:
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)
        print(f"\n[YOLO] 모델 로드 완료")

    def predict(self, image_path):
        # YOLO 추론 실행
        result = self.model.predict(
            source=image_path, conf=0.5, show=False, show_boxes=False, save=False
        )[0]

        names = self.model.names
        img_h, img_w = result.orig_shape
        total_area = img_h * img_w

        # 비율 및 카운트 초기화
        ratios = {"plastic": 0.0, "vinyl": 0.0, "wood": 0.0}
        count = 0

        # 원본 이미지 로드
        image = cv2.imread(image_path)

        # 마스크 및 박스 처리
        if result.masks is not None:
            for mask, box, cls_id, conf_val in zip(result.masks.data, result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                mask_np = mask.cpu().numpy()
                # 마스크 크기를 원본 이미지 크기로 리사이즈
                mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))

                class_id = int(cls_id)
                class_name = names[class_id]
                color = CLASS_COLORS.get(class_id, (255, 255, 255))  # 클래스 색상

                # 마스크 덧입히기
                mask_img = np.zeros_like(image, dtype=np.uint8)
                mask_img[mask_resized > 0.5] = color
                image = cv2.addWeighted(mask_img, ALPHA, image, 1 - ALPHA, 0)

                # 면적 비율 계산
                object_area = np.sum(mask_resized > 0.5)
                ratio = (object_area / total_area) * 100
                if class_name in ratios:
                    ratios[class_name] = round(ratios[class_name] + ratio, 2)

                # 🎯 바운딩 박스와 라벨 + 정확도 표시
                x1, y1, x2, y2 = map(int, box)
                conf = float(conf_val) * 100  # 정확도(%)
                # label = f"{class_name} {conf:.1f}%"
                label = f"{class_name}"
                # cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                count += 1

        # 결과 이미지 저장
        os.makedirs("temp", exist_ok=True)
        result_image_path = f"temp/result_{os.path.basename(image_path)}"
        cv2.imwrite(result_image_path, image)

        # 결과 이미지를 Base64로 변환
        with open(result_image_path, "rb") as f:
            result_img_base64 = base64.b64encode(f.read()).decode("utf-8")

        # 입력된 원본 이미지도 Base64 변환
        with open(image_path, "rb") as f:
            orig_img_base64 = base64.b64encode(f.read()).decode("utf-8")


        # 결과 딕셔너리 반환
        return {
            "orig_img": orig_img_base64,      # Base64 원본 이미지
            "rcnn_result": result_img_base64, # Base64 결과 이미지
            "wood": ratios["wood"],
            "plastic": ratios["plastic"],
            "vinyl": ratios["vinyl"],
            "count": count
        }


# 검출 면적 계산 근거
# 촬영 높이(H), 화각(HFOV/VFOV)이 고정이라면 → 바닥 면적은 "항상 동일"한 고정값
# 즉, 95.6m²는 고정된 상수 값으로 간주할 수 있고, 코드에서 매번 계산할 필요가 없습니다.
# 바닥면적 S = 2 * H * tan(HFOV / 2) * 2 * H * tan(VFOV / 2) = 95.6m²
# “1920×1080 해상도는 바닥 면적 계산에 직접적으로 쓰이지 않는다”는 점을 이해하는 것이 핵심입니다.
# 해상도는 “바닥 면적이 아닌 바닥 면적을 나누는 분해능”

# 개념 정리
# 촬영 높이 + 화각 : 카메라가 실제로 볼 수 있는 공간의 크기	→ 전체 바닥 면적(m²)을 결정
# 1920×1080 해상도 : 그 공간을 몇 개의 픽셀로 나눠서 표현하는지	→ 픽셀당 실제 면적 계산에 사용


# 계산 흐름
# [실제 바닥 전체 면적 95.6m²]
#      ↓ 해상도 1920×1080에 의해 픽셀로 분할
# [2,073,600 픽셀]
#      ↓ YOLO가 마스크 처리로 특정 객체 영역 픽셀 수 계산
# [예: object_pixels = 200,000]
#      ↓ 픽셀 비율
# pixel_ratio = 200,000 / 2,073,600 ≈ 0.096
#      ↓ 실제 면적 계산
# object_real_area = 95.6m² * 0.096 ≈ 9.18m²


# 실행 테스트용
if __name__ == "__main__":
    weight_path = "../weights/best.pt"
    image_path = "C:/Users/301/Desktop/create_image/image_4.png"

    yolo = YOLOWrapper(weight_path)
    result = yolo.predict(image_path)

    print("\n========== 🔍 분석 결과 ==========")
    print(f"[YOLO] 총 감지 객체 수: {result["count"]}")
    print(f"[YOLO] 나무 비율: {result['wood']}%")
    print(f"[YOLO] 플라스틱 비율: {result['plastic']}%")
    print(f"[YOLO] 비닐 비율: {result['vinyl']}%")
    print("==================================\n")

    # ✅ Base64 → OpenCV 이미지로 변환
    decoded = base64.b64decode(result['rcnn_result'])
    np_img = np.frombuffer(decoded, np.uint8)
    final_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    cv2.imshow("YOLO Detection Result", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()