# =====================================================
# 🧠 YOLOv8 Segmentation Training - Kaggle Auto Script
# =====================================================
#!pip install -q ultralytics

import os
import yaml
import shutil
from ultralytics import YOLO

# -----------------------------
# 1️⃣ Dataset 복사 (input → working)
# -----------------------------
input_dataset_dir = "C:/Users/301/Desktop/dataset"
working_dataset_dir = "C:/Users/301/Desktop/yolo_result2"


# # working 폴더 생성
# os.makedirs(working_dataset_dir, exist_ok=True)

# # train, val 폴더 복사
# for split in ["train", "val"]:
#     src_dir = os.path.join(input_dataset_dir, split)
#     dst_dir = os.path.join(working_dataset_dir, split)
#     if not os.path.exists(dst_dir):
#         shutil.copytree(src_dir, dst_dir)
#
# print("✅ train, val 폴더 복사 완료!")

# -----------------------------
# 2️⃣ data.yaml 새로 생성 (복사 X)
# -----------------------------

# yaml_path = os.path.join(working_dataset_dir, "data.yaml")

# data_config = {
#     "train": os.path.join(working_dataset_dir, "train/images"),
#     "val": os.path.join(working_dataset_dir, "val/images"),
#     "nc": 3,
#     "names": ["wood", "vinyl", "glass"]
# }

# ✅ Rewrite YAML file (상대경로 + 깔끔한 구조)
yaml_content = {
    "train": "C:/Users/301/Desktop/dataset/train/images",
    "val": "C:/Users/301/Desktop/dataset/val/images",
    "nc": 3,
    "names": ["wood", "vinyl", "plastic"]
}
yaml_path = os.path.join(working_dataset_dir, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(yaml_content, f)
print(f"✅ data.yaml 재작성 완료! 경로: {yaml_path}")

print("📄 YAML 내용:")
print(yaml.dump(yaml_content, sort_keys=False))


# -----------------------------
# 3️⃣ YOLOv8 세그멘테이션 학습
# -----------------------------
model = YOLO("yolov8n-seg.pt")

# model.train(
#     data=yaml_path,          # working 폴더의 새로 만든 yaml 파일로 학습
#     task="segment",
#     epochs=50,
#     imgsz=640,
#     batch=4,
#     project="C:/Users/dayeon/Desktop/yolo_result/runs",
#     name="yolo_seg_train",
#     amp=False,   # 🚫 AMP 비활성화
#     workers=0   # 👈 DataLoader 문제 방지
# )
model.train(
    data=yaml_path,          # 기존 data.yaml
    task="segment",
    epochs=50,               # 🔹 에폭 줄임
    imgsz=640,
    batch=8,                 # 🔹 배치 늘림
    project="C:/Users/301/Desktop/yolo_result/runs",
    name="yolo_seg_train",   # 기존 경로 덮어쓰기
    amp=True,                # 🔹 GPU에서는 Mixed Precision 활성화
    workers=2                # 🔹 DataLoader 안정적 설정
)

print("\n🎉 학습 완료! 최종 모델 위치:")
print("C:/Users/301/Desktop/yolo_result/runs/yolo_seg_train/weights/best.pt")