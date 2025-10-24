import os
import numpy as np
import joblib
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Windows라면 기본적으로 'Malgun Gothic'이 설치되어 있음
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# -------------------------------
# 1. 기반 PCA 모델 및 데이터 불러오기
# -------------------------------
pca = joblib.load("models/pca_model.pkl")
explained = np.load("models/pca_explained.npy")
#features = np.load("models/features.npy")      # 기준 데이터의 비율 벡터
features_std = np.load("models/features_std.npy")   # 표준화된 특징
labels = np.load("models/labels.npy")          # 기준 데이터 클래스 라벨

#pca_result = pca.transform(features)            # 기준 데이터의 PCA 결과
pca_result = pca.transform(features_std)            # 기준 데이터의 PCA 결과

model = YOLO("../weights/best.pt")

CLASS_NAMES = {0: 'wood', 1: 'vinyl', 2: 'plastic'}
CLASS_LIST = ['wood', 'vinyl', 'plastic']
colors = {'wood': 'blue', 'vinyl': 'orange', 'plastic': 'green'}

# -------------------------------
# 2. 사용자 이미지 특징 추출 함수
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
            cls_name = CLASS_NAMES[int(cls)]
            class_areas[cls_name] += np.sum(mask.cpu().numpy().astype(np.uint8))

    return [
        class_areas['wood'] / total_area,
        class_areas['vinyl'] / total_area,
        class_areas['plastic'] / total_area
    ]

# -------------------------------
# 3. 사용자 이미지 분석
# -------------------------------
user_image_path = "C:/Users/301/Desktop/create_image/image_yolo30.png"
user_ratios = extract_ratios(user_image_path)
user_pca = pca.transform([user_ratios])[0]

print("\n⭐ 사용자 이미지 PCA 좌표:", user_pca)

# -------------------------------
# 4. PCA 축 해석 출력
# -------------------------------
components = pca.components_
pc1_desc = np.argmax(np.abs(components[0]))
pc2_desc = np.argmax(np.abs(components[1]))

print("\n📌 PCA 축 해석")
print(f"PC1은 '{CLASS_LIST[pc1_desc]}' 비율에 가장 큰 영향을 받습니다 (가중치: {components[0][pc1_desc]:.4f})")
print(f"PC2는 '{CLASS_LIST[pc2_desc]}' 비율에 가장 큰 영향을 받습니다 (가중치: {components[1][pc2_desc]:.4f})")

# -------------------------------
# 5. 전체 데이터 산점도 + 사용자 이미지 표시
# -------------------------------
plt.figure(figsize=(8, 6))

for cls in CLASS_LIST:
    idx = np.where(labels == cls)[0]
    plt.scatter(pca_result[idx, 0], pca_result[idx, 1], color=colors[cls], label=cls, alpha=0.6)

# 사용자 이미지 표시
plt.scatter(user_pca[0], user_pca[1], color='red', marker='*', s=250, label='User Image')

plt.title("PCA Scatter: Full Dataset and User Image")
plt.xlabel(f"PC1 ({explained[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}%)")
plt.legend()
plt.grid(True)
plt.show()
