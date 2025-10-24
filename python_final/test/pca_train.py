import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.decomposition import PCA


# YOLO 모델 로드
class YOLOWrapper:
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)
        self.class_names = self.model.names  # 예: {0:'plastic',1:'vinyl',2:'wood'}

    def extract_ratios(self, image_path):
        result = self.model.predict(source=image_path, conf=0.25, show=False, boxes=False)[0]
        img_h, img_w = result.orig_shape
        total_area = img_h * img_w

        ratios = {"plastic": 0.0, "vinyl": 0.0, "wood": 0.0}
        if result.masks is not None:
            for mask, cls_id in zip(result.masks.data, result.boxes.cls):
                class_name = self.class_names[int(cls_id)]
                mask_np = mask.cpu().numpy()
                object_area = np.sum(mask_np > 0.5)
                ratio = (object_area / total_area) * 100
                if class_name in ratios:
                    ratios[class_name] += ratio
        return ratios["plastic"], ratios["vinyl"], ratios["wood"]


# PCA 학습 수행
def train_pca(data_dir, yolo_model_path, pca_save_path="pca_model.pkl"):
    yolo = YOLOWrapper(yolo_model_path)

    X = []  # PCA 입력 데이터
    y = []  # 클래스 라벨
    label_map = {"plastic": 0, "vinyl": 1, "wood": 2}

    for class_name in label_map.keys():
        class_path = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(class_path, img_file)
            try:
                plastic, vinyl, wood = yolo.extract_ratios(img_path)
                X.append([plastic, vinyl, wood])
                y.append(label_map[class_name])
                print(f"[OK] {img_path} → {plastic:.2f}, {vinyl:.2f}, {wood:.2f}")
            except Exception as e:
                print(f"[ERROR] {img_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    # PCA 학습
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    joblib.dump(pca, pca_save_path)
    print(f"✅ PCA 모델 저장 완료: {pca_save_path}")

    # 산점도 시각화
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green']
    labels = ['plastic', 'vinyl', 'wood']

    for label, color, name in zip([0, 1, 2], colors, labels):
        idx = y == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], c=color, label=name)

    plt.title('PCA Result Based on YOLO Ratios')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

    return pca


# 실행 부분
if __name__ == "__main__":
    DATASET_DIR = r"C:/your_dataset_path"  # 🔥 폴더 경로 수정
    YOLO_MODEL_PATH = r"C:/Users/301/Desktop/final_result80/runs/yolo_seg_train/weights/best.pt"

    train_pca(DATASET_DIR, YOLO_MODEL_PATH)
