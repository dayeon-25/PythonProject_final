from ultralytics import YOLO
import cv2, os, numpy as np

# 🎨 전역 색상 설정 (BGR 형식)
CLASS_COLORS = {
    0: (0, 255, 0),     # 나무 → 초록
    1: (0, 255, 255),   # 플라스틱 → 노랑
    2: (0, 0, 255)      # 비닐 → 빨강
}

ALPHA = 0.5  # 마스크 투명도 설정 (0~1)

class YOLOWrapper:
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)
        #print(f"[YOLO] ✅ 모델 로드 완료: {weight_path}")
        print(f"[YOLO] ✅ 모델 로드 완료")


    def predict(self, image_path):
        # 🔍 YOLO 추론
        result = self.model.predict(
            source=image_path, conf=0.5, show=False, show_boxes=False, save=False
        )[0]

        names = self.model.names
        img_h, img_w = result.orig_shape
        total_area = img_h * img_w

        # 📊 비율 및 카운트 초기화
        ratios = {"plastic": 0.0, "vinyl": 0.0, "wood": 0.0}
        count = 0

        # 📌 원본 이미지 로드
        image = cv2.imread(image_path)

        # 🎭 마스크 및 박스 처리
        if result.masks is not None:
            for mask, box, cls_id, conf_val in zip(result.masks.data, result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                mask_np = mask.cpu().numpy()
                # ✅ 마스크 크기를 원본 이미지 크기로 리사이즈
                mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))

                class_id = int(cls_id)
                color = CLASS_COLORS.get(class_id, (255, 255, 255))  # 클래스 색상 가져오기

                # 🔴 세그멘테이션 마스크 적용
                mask_img = np.zeros_like(image, dtype=np.uint8)
                #mask_img[mask_np > 0.5] = color
                mask_img[mask_resized > 0.5] = color
                image = cv2.addWeighted(mask_img, ALPHA, image, 1 - ALPHA, 0)

                # 📌 면적 계산 및 누적
                #object_area = np.sum(mask_np > 0.5)
                object_area = np.sum(mask_resized > 0.5)

                ratio = (object_area / total_area) * 100

                class_name = names[class_id]
                if class_name in ratios:
                    ratios[class_name] = round(ratios[class_name] + ratio, 2)

                # 🎯 바운딩 박스와 라벨 + 정확도 표시
                x1, y1, x2, y2 = map(int, box)
                conf = float(conf_val) * 100  # 정확도(%)
                label = f"{class_name} {conf:.1f}%"
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                count += 1

        # 🖼️ 결과 이미지 저장 및 출력
        os.makedirs("temp", exist_ok=True)
        result_image_path = f"temp/result_{os.path.basename(image_path)}"
        cv2.imwrite(result_image_path, image)

        print("\n========== 🔍 분석 결과 ==========")
        print(f"총 감지 객체 수: {count}")
        print(f"나무 비율: {ratios['wood']}%")
        print(f"플라스틱 비율: {ratios['plastic']}%")
        print(f"비닐 비율: {ratios['vinyl']}%")
        print(f"결과 이미지 저장 위치: {result_image_path}")
        print("==================================\n")

        # 결과 이미지 표시
        cv2.imshow("YOLO Detection Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 사용자 설정
    # weight_path = "best.pt"  # 📌 자신의 모델 경로 입력
    # image_path = "test_image.jpg"  # 📌 테스트할 이미지 경로
    # 모델 로드
    weight_path = YOLO("C:/Users/301/Desktop/final_result80/runs/yolo_seg_train/weights/best.pt")
    # 이미지 경로
    image_path = "C:/Users/301/Desktop/create_image/goljae.png"

    yolo = YOLOWrapper(weight_path)
    yolo.predict(image_path)
