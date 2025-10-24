from ultralytics import YOLO
import numpy as np
import cv2
import base64
import os


# 패키지 설치
# pip install ultralytics



class YOLOWrapper:
    def __init__(self, weight_path="C:/Users/301/Desktop/final_result80/runs/yolo_seg_train/weights/best.pt"):
        """
        YOLOWrapper 초기화
        :param weight_path: 학습된 모델 weight 파일 경로
        """
        self.weight_path = weight_path
        self.model = YOLO(weight_path)
        self.class_names = self.model.names  # 클래스 이름 리스트

    def _apply_mask_color(self, img, masks, classes):
        """
        객체별로 다른 색을 적용한 마스크 이미지 생성
        """
        seg_img = img.copy()
        rng = np.random.default_rng(42)  # 고정된 색상 생성용 (재현성 확보)
        color_map = {}

        h, w, _ = img.shape  # 원본 이미지 크기

        for i, mask in enumerate(masks):
            cls_id = int(classes[i])
            # 클래스별 색상이 고정되도록
            if cls_id not in color_map:
                color_map[cls_id] = rng.integers(0, 256, size=3).tolist()
            color = color_map[cls_id]

            # ✅ 마스크를 원본 이미지 크기로 resize
            # ✅ torch tensor → numpy 변환 후 resize
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask.cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)

            # ✅ 크기 확인 (디버그용)
            print(mask_resized.shape, img.shape)

            seg_img[mask > 0.5] = (
                    seg_img[mask > 0.5] * 0.5 + np.array(color) * 0.5
            )

        return seg_img


    def _calculate_area_ratios(self, masks, img_shape):
        """
         각 객체가 이미지에서 차지하는 비율 계산 (%)
        """
        h, w, _ = img_shape
        total_area = h * w
        ratios = []

        for mask in masks:
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            area = np.sum(mask_resized > 0.5)
            ratio = (area / total_area) * 100
            ratios.append(ratio)

        return ratios

    def _count_objects(self, classes):
        """
        인식된 객체의 수 반환
        """
        return len(classes)

    def predict(self, img):
        """
        YOLOv8 세그멘테이션 예측 (색상, 비율, 객체 수 포함)
        """
        results = self.model(img)
        masks = results[0].masks.data
        classes = [int(cls) for cls in results[0].boxes.cls]

        # 1️⃣ 객체별 색상 입힌 결과 이미지
        seg_img = self._apply_mask_color(img, masks, classes)

        # 2️⃣ 객체별 면적 비율 계산
        ratios = self._calculate_area_ratios(masks, img.shape)

        # 3️⃣ 인식된 객체 수 계산
        num_objects = self._count_objects(classes)

        # 객체 이름 매핑
        class_names = [self.class_names[c] for c in classes]

        return seg_img, list(zip(class_names, ratios)), num_objects



    # def train(self, data_path, epochs=50, imgsz=640, save_path=None):
    #     """
    #     YOLO 학습
    #     :param data_path: 학습 데이터 경로 (YOLO 형식)
    #     :param epochs: 학습 epoch
    #     :param imgsz: 이미지 크기
    #     :param save_path: 학습 후 저장할 weight 파일
    #     """
    #     self.model.train(data=data_path, epochs=epochs, imgsz=imgsz)
    #     if save_path:
    #         self.model.save(save_path)
    #         self.weight_path = save_path
    #     print(f"YOLO 학습 완료, 모델 경로: {self.weight_path}")