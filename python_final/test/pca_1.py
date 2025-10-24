# import os
# import numpy as np
# import cv2
# from ultralytics import YOLO
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
#
# # -------------------------------
# # 1. YOLO 모델 로드
# # -------------------------------
# model = YOLO("../weights/best.pt")  # 모델 경로 맞춰주세요
#
# # 클래스 이름 정의 (사용자 제공 순서 반영)
# CLASS_NAMES = {0: 'wood', 1: 'vinyl', 2: 'plastic'}
#
# # -------------------------------
# # 2. YOLO 특징 추출 함수
# # -------------------------------
# def extract_ratios(image_path):
#     results = model(image_path)
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"❌ 이미지 로드 실패: {image_path}")
#         return None
#     h, w, _ = img.shape
#     total_area = h * w
#
#     class_areas = {'wood': 0, 'vinyl': 0, 'plastic': 0}
#
#     for result in results:
#         masks = result.masks
#         if masks is None:
#             continue
#         for mask, cls in zip(masks.data, result.boxes.cls):
#             cls_name = CLASS_NAMES[int(cls)]
#             mask_np = mask.cpu().numpy().astype(np.uint8)
#             area = np.sum(mask_np)
#             class_areas[cls_name] += area
#
#     return [
#         class_areas['wood'] / total_area,
#         class_areas['vinyl'] / total_area,
#         class_areas['plastic'] / total_area
#     ]
#
# # -------------------------------
# # 3. 기준 데이터셋 로드
# # -------------------------------
# dataset_dir = "../image"
# features = []
# labels = []
# file_names = []
#
# for cls_index, cls_name in CLASS_NAMES.items():
#     class_folder = os.path.join(dataset_dir, cls_name)
#     for filename in os.listdir(class_folder):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             img_path = os.path.join(class_folder, filename)
#             ratios = extract_ratios(img_path)
#             if ratios:
#                 features.append(ratios)
#                 labels.append(cls_name)
#                 file_names.append(filename)
#
# features = np.array(features)
# print("✅ 기준 데이터 특징 추출 완료:", features.shape)
#
# # -------------------------------
# # 4. PCA 학습
# # -------------------------------
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(features)
# print("📌 PCA 학습 완료:", pca_result.shape)
#
# # -------------------------------
# # 5. 사용자 이미지 분석
# # -------------------------------
# user_image_path = "C:/Users/301/Desktop/ChatGPT_ Image2.png"
# user_ratios = extract_ratios(user_image_path)
# user_pca = pca.transform([user_ratios])[0]  # 2D 좌표
#
# # -------------------------------
# # 6. 산점도 시각화
# # -------------------------------
# plt.figure(figsize=(8, 6))
#
# # 기준 데이터 산점도
# for cls in CLASS_NAMES.values():
#     idx = [i for i, label in enumerate(labels) if label == cls]
#     plt.scatter(pca_result[idx, 0], pca_result[idx, 1], label=cls, alpha=0.7)
#
# # 사용자 이미지 표시
# plt.scatter(user_pca[0], user_pca[1], color='red', marker='*', s=200, label='User Image')
# plt.title("PCA Scatter: User Image Position")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.legend()
# plt.grid(True)
# plt.show()




# 문구 출력 추가 버전
import os
import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -------------------------------
# 1. YOLO 모델 로드
# -------------------------------
model = YOLO("../weights/best.pt")  # 모델 경로 맞춰주세요

# 클래스 이름 정의 (사용자 제공 순서 반영)
CLASS_NAMES = {0: 'wood', 1: 'vinyl', 2: 'plastic'}  # 키: 클래스 인덱스, 값: 클래스명
CLASS_LIST = ['wood', 'vinyl', 'plastic']  # 순서를 유지한 리스트

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

    class_areas = {'wood': 0, 'vinyl': 0, 'plastic': 0}

    for result in results:
        masks = result.masks
        if masks is None:
            continue
        for mask, cls in zip(masks.data, result.boxes.cls):
            cls_name = CLASS_NAMES[int(cls)]
            mask_np = mask.cpu().numpy().astype(np.uint8)
            area = np.sum(mask_np)
            class_areas[cls_name] += area

    return [
        class_areas['wood'] / total_area,
        class_areas['vinyl'] / total_area,
        class_areas['plastic'] / total_area
    ]

# -------------------------------
# 3. 기준 데이터셋 로드
# -------------------------------
dataset_dir = "../image"  # dataset 폴더 경로
features = []
labels = []
file_names = []

for cls_index, cls_name in CLASS_NAMES.items():
    class_folder = os.path.join(dataset_dir, cls_name)
    if not os.path.exists(class_folder):
        print(f"❌ 폴더가 존재하지 않습니다: {class_folder}")
        continue

    for filename in os.listdir(class_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(class_folder, filename)
            ratios = extract_ratios(img_path)
            if ratios:
                features.append(ratios)
                labels.append(cls_name)
                file_names.append(filename)

features = np.array(features) # 18초 걸림
print("✅ 기준 데이터 특징 추출 완료:", features.shape)

# -------------------------------
# 4. PCA 학습
# -------------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)
explained = pca.explained_variance_ratio_

print(f"📌 PCA 설명 분산 비율: PC1={explained[0]:.4f}, PC2={explained[1]:.4f}")

# -------------------------------
# 4-2. PCA 축의 의미 자동 분석
# -------------------------------
components = pca.components_
pc1_weights = components[0]
pc2_weights = components[1]

def interpret_pc(weights, name):
    idx = np.argmax(np.abs(weights))
    feature_name = CLASS_LIST[idx]
    influence = weights[idx]
    direction = "양의 방향" if influence > 0 else "음의 방향"
    return f"{name}는 '{feature_name}' 비율의 {direction} 변화에 가장 크게 영향을 받습니다 (가중치 {influence:.4f})."

pc1_desc = interpret_pc(pc1_weights, "PC1")
pc2_desc = interpret_pc(pc2_weights, "PC2")

print("\n📌 PCA 축 해석")
print(pc1_desc)
print(pc2_desc)

# -------------------------------
# 5. 사용자 이미지 분석
# -------------------------------
user_image_path = "C:/Users/301/Desktop/ChatGPT_ Image2.png" # 실제 경로에 맞게 수정
user_ratios = extract_ratios(user_image_path)
user_pca = pca.transform([user_ratios])[0]  # 2D 좌표

# -------------------------------
# 5-2. 사용자 이미지가 가장 가까운 클래스 판별
# -------------------------------
# 각 클래스별 PCA 좌표의 평균 계산
class_means = {}
for cls_name in CLASS_LIST:
    idx = [i for i, label in enumerate(labels) if label == cls_name]
    class_means[cls_name] = np.mean(pca_result[idx], axis=0)

distances = {cls: np.linalg.norm(user_pca - mean) for cls, mean in class_means.items()}
closest_class = min(distances, key=distances.get)

print("\n⭐ 사용자 이미지 분석 결과:")
print(f"📍 PCA 좌표: {user_pca}")
print(f"🎯 가장 가까운 클래스는 '{closest_class}' 입니다.")
print("📏 클래스별 거리:", distances)

# -------------------------------
# 6. 산점도 시각화
# -------------------------------
plt.figure(figsize=(8, 6))

for cls in CLASS_LIST:
    idx = [i for i, label in enumerate(labels) if label == cls]
    plt.scatter(pca_result[idx, 0], pca_result[idx, 1], label=cls, alpha=0.7)

plt.scatter(user_pca[0], user_pca[1], color='red', marker='*', s=200, label='User Image')
plt.title("PCA Scatter: User Image Position")
plt.xlabel(f"PC1 ({explained[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}%)")
plt.legend()
plt.grid(True)
plt.show()