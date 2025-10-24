# from ultralytics import YOLO
# import cv2
# import matplotlib.pyplot as plt
# from pathlib import Path
# import numpy as np
#
# # 모델 로드
# #model = YOLO("C:/Users/301/Desktop/yolo_result2/runs/yolo_seg_train/weights/best.pt")  # 학습된 모델 경로
# #model = YOLO("C:/Users/301/Desktop/yolo_result_wood/runs/yolo_seg_train2/weights/best.pt")  # 학습된 모델 경로
# #model = YOLO("C:/Users/301/Desktop/final_result/runs/yolo_seg_train5/weights/best.pt")
# #model = YOLO("C:/Users/301/Desktop/final_result20/runs/yolo_seg_train/weights/best.pt")
# model = YOLO("C:/Users/301/Desktop/final_result80/runs/yolo_seg_train/weights/best.pt")
#
# # 이미지 경로
# #image_path = "C:/Users/301/Desktop/636962_516882_4909.jpg"  # 원하는 이미지 파일 경로로 변경
# #image_path = "C:/Users/301/Desktop/create_image/image_17.png"  # 원하는 이미지 파일 경로로 변경
# #image_path = "C:/Users/301/PycharmProjects/PythonProject_final/image/vinyl/15_X001_C042_0923_0.jpg"
# #image_path = "C:/Users/301/PycharmProjects/PythonProject_final/image/plastic/20240711_151524_jpg.rf.eae34ecca3bb2acf0c34a6fe27b29bb6.jpg"
# #image_path = "C:/Users/301/PycharmProjects/PythonProject_final/image/wood/81_jpg.rf.7c0686e83be1344b6394ea9c12c13b69.jpg"
# image_path = "C:/Users/301/Desktop/create_image/image_yolo7.png"
#
# # 🎯 클래스별 색상 정의 (BGR 형식)
# CLASS_COLORS = {
#     0: (0, 255, 0),     # 나무 → 초록
#     1: (0, 255, 255),   # 플라스틱 → 노랑
#     2: (0, 0, 255)      # 비닐 → 빨강
# }
#
# # 예측 수행
# results = model.predict(
#     source=image_path,
#     show=False,
#     conf=0.8,
#     save=False,
#     show_boxes=False,
#     save_txt=True,
#     save_conf=True
# )
#
# # 원본 이미지 불러오기
# orig_img = cv2.imread(image_path)
# if orig_img is None:
#     print("❌ 이미지 불러오기 실패 — 경로 또는 파일 문제")
#     exit()
#
#
# img_h, img_w = orig_img.shape[:2]
# total_area = img_h * img_w
#
#
# # 결과 이미지 경로
# #result = results[0].plot(boxes=False)
# #result_image_path = Path(result.save_dir) / Path(result.path).name
# result_image_path = Path(results[0].save_dir) / Path(results[0].path).name
# print("✅ 탐지된 결과 이미지 경로:", result_image_path)
#
#
# for r in results:
#     print("r 확인")
#     print(r)
#     img_h, img_w = r.orig_shape  # 원본 이미지 크기
#     total_area = img_h * img_w   # 전체 픽셀 수
#
#     # 결과 반복
#     for i, (mask, cls_id) in enumerate(zip(r.masks.data, r.boxes.cls)):
#         mask_np = mask.cpu().numpy()
#
#         # 객체 픽셀 수 계산
#         object_area = np.sum(mask_np > 0.5)
#         ratio = (object_area / total_area) * 100
#
#         # 클래스 이름 가져오기
#         class_id = int(cls_id)
#         class_name = model.names[class_id]  # YOLOv8이 자동 저장함
#
#         print(f"객체 {i+1}: {class_name} ({class_id}) → {ratio:.2f}% 차지")
#
#
# # 시각화
# # img = cv2.imread(str(result_image_path))
# # if img is None:
# #     print("❌ 이미지 불러오기 실패 — 경로 또는 파일 문제")
# # else:
# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #     plt.imshow(img)
# #     plt.axis("off")
# #     plt.title("YOLO Detection Result")
# #     plt.show()


from ultralytics import YOLO
import cv2, os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 🎨 클래스별 색상 정의 (BGR)
CLASS_COLORS = {
    0: (0, 255, 0),     # 나무 → 초록
    1: (0, 255, 255),   # 플라스틱 → 노랑
    2: (0, 0, 255)      # 비닐 → 빨강
}

ALPHA = 0.5  # 마스크 투명도 (0~1)

# 모델 로드
model = YOLO("C:/Users/301/Desktop/final_result80/runs/yolo_seg_train/weights/best.pt")

# 이미지 경로
image_path = "C:/Users/301/Desktop/create_image/goljae.png"

# 예측 수행
results = model.predict(
    source=image_path,
    conf=0.8,
    save=False,     # 결과 저장은 아래에서 별도 처리
    show_boxes=True,     # 박스 포함
    show=False
)

# 원본 이미지 로드
orig_img = cv2.imread(image_path)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)  # plt 표시용 (RGB 변환)
img_h, img_w = orig_img.shape[:2]

# 결과 저장 위치 가져오기 (YOLO가 자동 지정한 폴더)
#save_dir = Path(results[0].save_dir)  # 예: runs/segment/predict5
#print(f"📂 라벨과 이미지가 저장될 경로: {save_dir}")

# result_image_path = Path(results[0].save_dir) / Path(results[0].path).name
# print(f"📂 라벨과 이미지가 저장될 경로: {result_image_path}")


# 결과 처리
for r in results:
    masks = r.masks.data if r.masks is not None else []
    classes = r.boxes.cls if r.boxes is not None else []
    boxes = r.boxes.xyxy if r.boxes is not None else []
    scores = r.boxes.conf if r.boxes is not None else []

    for mask, cls_id, box, score in zip(masks, classes, boxes, scores):
        mask_np = mask.cpu().numpy()
        cls_id = int(cls_id)

        # 🎯 마스크를 원본 이미지 크기에 맞게 리사이즈
        mask_resized = cv2.resize(mask_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        mask_binary = mask_resized > 0.5

        # 🎨 마스크 색 적용
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))  # 기본 흰색
        colored_mask = np.zeros_like(orig_img, dtype=np.uint8)
        colored_mask[:, :] = color

        # ✅ Alpha Blending 적용
        orig_img = np.where(
            mask_binary[..., None],
            (ALPHA * colored_mask + (1 - ALPHA) * orig_img).astype(np.uint8),
            orig_img
        )

        # ✅ Bounding Box 좌표 추출
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        #cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, 2)

        # ✅ 클래스 라벨 + confidence 표시
        #label = f"{model.names[int(cls_id)]}: {score:.2f}"
        label = f"{model.names[int(cls_id)]}"
        cv2.putText(orig_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# YOLO 결과 저장 디렉토리 확인
#save_dir = Path(results[0].save_dir)  # 라벨이 저장된 폴더 경로

# 저장 경로 지정
# #output_path = result_image_path / "output_final_visual.png"  # 저장할 이미지 경로 지정orig_img_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
# output_path = Path(results[0].save_dir) / Path(results[0].path).name
# print(f"📂 라벨과 이미지가 저장될 경로: {output_path}")

# YOLO 결과 이미지 저장
os.makedirs("temp", exist_ok=True)
result_image_path = f"temp/result_{os.path.basename(image_path)}"
#results.save(filename=result_image_path)

# 결과 저장 (BGR 변환 후 저장)
orig_img_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(result_image_path), orig_img_bgr)
print(f"✅ 최종 시각화 이미지 저장 완료: {result_image_path}")

# 결과 출력
plt.figure(figsize=(10, 8))
plt.imshow(orig_img)
plt.title("YOLO Segmentation + Transparent Mask + Boxes + Labels")
plt.axis("off")
plt.show()

