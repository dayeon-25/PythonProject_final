import os
import numpy as np
import joblib
import cv2
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO
from sklearn.decomposition import PCA

# https://doing-nothing.tistory.com/79 pca 분석 정규화 과정 참고


# -------------------------------
# 1. YOLO 모델 로드
# -------------------------------
model = YOLO("../weights/best.pt")

# 클래스 이름 정의
CLASS_NAMES = {0: 'wood', 1: 'vinyl', 2: 'plastic'}
CLASS_LIST = ['wood', 'vinyl', 'plastic']

# -------------------------------
# 2. YOLO 특징 추출 함수
# -------------------------------
def extract_ratios(image_path):
    results = model(image_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 이미지 로드 실패: {image_path}")
        return None
    h, w, _ = img.shape
    total_area = h * w

    class_areas = {name: 0 for name in CLASS_LIST}

    for result in results:
        masks = result.masks
        if masks is None:
            continue
        for mask, cls in zip(masks.data, result.boxes.cls):
            print(mask.shape, cls)
            cls_name = CLASS_NAMES[int(cls)]
            mask_np = mask.cpu().numpy().astype(np.uint8)
            class_areas[cls_name] += np.sum(mask_np)

    return [
        class_areas['wood'] / total_area,
        class_areas['vinyl'] / total_area,
        class_areas['plastic'] / total_area
    ]

# -------------------------------
# 3. 기준 데이터셋 로드 및 특징 추출
# -------------------------------
dataset_dir = "../image"
features = []
labels = []

for cls_index, cls_name in CLASS_NAMES.items():
    class_folder = os.path.join(dataset_dir, cls_name)
    for filename in os.listdir(class_folder):
        if filename.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(class_folder, filename)
            ratios = extract_ratios(img_path)
            if ratios:
                features.append(ratios)
                labels.append(cls_name)

features = np.array(features)
labels = np.array(labels)

print("✅ 특징 추출 완료:", features.shape)

# # -------------------------------
# # 4. PCA 학습 및 저장
# # -------------------------------
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(features)
# explained = pca.explained_variance_ratio_
# print(f"📌 PCA 설명된 분산 비율: {explained}")
#
# # 클래스별 평균 (거리 계산용)
# # class_means = {}
# # for cls in CLASS_LIST:
# #     idx = np.where(labels == cls)[0]
# #     class_means[cls] = np.mean(pca_result[idx], axis=0)
#
# # 저장
# os.makedirs("models", exist_ok=True)
# joblib.dump(pca, "models/pca_model.pkl")
# # np.save("models/class_means.npy", class_means)
# np.save("models/features.npy", features)
# np.save("models/labels.npy", labels)
# np.save("models/pca_explained.npy", explained)
# print("✅ PCA 모델 및 클래스 중심 저장 완료!")

# -------------------------------
# 4. 표준화 + PCA 학습
# -------------------------------
scaler = StandardScaler()
features_std = scaler.fit_transform(features)   # 표준화 적용

pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_std)

explained = pca.explained_variance_ratio_
print(f"📌 PCA 설명된 분산 비율 (Standardized): {explained}")

# -------------------------------
# 5. 저장
# -------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(pca, "models/pca_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")   # 추가: scaler 저장
np.save("models/features.npy", features)
np.save("models/features_std.npy", features_std)
np.save("models/labels.npy", labels)
np.save("models/pca_explained.npy", explained)

print("✅ 표준화 + PCA 모델 저장 완료!")