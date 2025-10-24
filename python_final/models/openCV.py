# import cv2
# import numpy as np
# import base64
#
# class OpenCVWrapper:
#     def __init__(self):
#         # HSV 색상 범위 (명도, 채도 낮게잡고 범위 넓게)
#         self.color_ranges = {
#             "plastic": [  # 병뚜껑 - 빨강, 파랑, 초록
#                 {"lower": (0, 80, 80), "upper": (10, 255, 255)},   # 빨강
#                 {"lower": (170, 80, 80), "upper": (180, 255, 255)},# 빨강2
#                 {"lower": (100, 80, 80), "upper": (130, 255, 255)},# 파랑
#                 {"lower": (40, 50, 50), "upper": (90, 255, 255)}   # 초록
#             ],
#             "vinyl": [  # 검정색 비닐
#                 {"lower": (0, 0, 0), "upper": (180, 255, 60)}      # 매우 어두운 영역
#             ],
#             "wood": [   # 갈색 계열
#                 {"lower": (10, 50, 50), "upper": (30, 255, 255)}   # 노랑~갈색 범위
#             ]
#         }
#
#         # 시각화를 위한 색상 지정 (BGR)
#         self.visual_colors = {
#             "plastic": (255, 0, 0),   # Blue
#             "vinyl": (0, 0, 255),     # Red
#             "wood": (0, 255, 255)     # Yellow
#         }
#
#     def detect_all_in_one(self, image_path):
#         img = cv2.imread(image_path)
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#         overlay = img.copy()
#
#         for material, ranges in self.color_ranges.items():
#             mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
#             for r in ranges:
#                 mask = cv2.inRange(hsv, r["lower"], r["upper"])
#                 mask_total = cv2.bitwise_or(mask_total, mask)
#
#             # 모폴로지 연산으로 잡음 제거
#             kernel = np.ones((5, 5), np.uint8)
#             mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
#             mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)
#
#             # 컬러 오버레이
#             color = self.visual_colors[material]
#             colored_mask = np.zeros_like(img)
#             colored_mask[:, :] = color
#
#             overlay = cv2.addWeighted(overlay, 1, cv2.bitwise_and(colored_mask, colored_mask, mask=mask_total), 0.5, 0)
#
#         # Base64로 변환
#         _, buffer = cv2.imencode('.png', overlay)
#         encoded_image = base64.b64encode(buffer).decode('utf-8')
#         return encoded_image



# # 이미지에 색상별 라벨 까지 포함된 버전
# import cv2
# import numpy as np
# import base64
# import os
#
# class OpenCVWrapper:
#     def __init__(self):
#         # HSV 색상 범위
#         self.color_ranges = {
#             "plastic": [
#                 {"lower": (0, 80, 80), "upper": (10, 255, 255)},   # 빨강
#                 {"lower": (170, 80, 80), "upper": (180, 255, 255)},
#                 {"lower": (100, 80, 80), "upper": (130, 255, 255)},# 파랑
#                 {"lower": (40, 50, 50), "upper": (90, 255, 255)}   # 초록
#             ],
#             "vinyl": [
#                 {"lower": (0, 0, 0), "upper": (180, 255, 60)}      # 검정
#             ],
#             "wood": [
#                 {"lower": (10, 50, 50), "upper": (30, 255, 255)}   # 갈색 계열
#             ]
#         }
#
#         # 시각화 색상 (BGR)
#         self.visual_colors = {
#             "plastic": (255, 0, 0),   # Blue
#             "vinyl": (0, 0, 255),     # Red
#             "wood": (0, 255, 255)     # Yellow
#         }
#
#     def detect_all_in_one(self, image_path):
#         img = cv2.imread(image_path)
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         overlay = img.copy()
#
#         for material, ranges in self.color_ranges.items():
#             mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
#             for r in ranges:
#                 mask = cv2.inRange(hsv, r["lower"], r["upper"])
#                 mask_total = cv2.bitwise_or(mask_total, mask)
#
#             kernel = np.ones((5, 5), np.uint8)
#             mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
#             mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)
#
#             color = self.visual_colors[material]
#             colored_mask = np.zeros_like(img)
#             colored_mask[:] = color
#             overlay = cv2.addWeighted(
#                 overlay, 1, cv2.bitwise_and(colored_mask, colored_mask, mask=mask_total), 0.5, 0
#             )
#
#         # ✅ 범례 추가
#         legend_img = self.add_legend(overlay)
#
#         # ✅ Base64 변환
#         _, buffer = cv2.imencode('.png', legend_img)
#         encoded_image = base64.b64encode(buffer).decode('utf-8')
#         return encoded_image
#
#     def add_legend(self, image):
#         legend_height = 50
#         legend = np.zeros((legend_height, image.shape[1], 3), dtype=np.uint8)
#
#         x_offset = 10
#         for material, color in self.visual_colors.items():
#             # 박스 그리기
#             cv2.rectangle(legend, (x_offset, 10), (x_offset + 30, 40), color, -1)
#             # 텍스트 넣기
#             label = material.upper()
#             cv2.putText(legend, label, (x_offset + 40, 35),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#             x_offset += 150
#
#         # ✅ 이미지와 범례 결합
#         return np.vstack((image, legend))





# 과정 이미지까지 포함된 버전
import cv2
import numpy as np
import base64
from io import BytesIO

class OpenCVWrapper:
    def __init__(self):
        self.color_ranges = {
            "plastic": [
                {"lower": (0, 80, 80), "upper": (10, 255, 255)},
                {"lower": (170, 80, 80), "upper": (180, 255, 255)},
                {"lower": (100, 80, 80), "upper": (130, 255, 255)},
                {"lower": (40, 50, 50), "upper": (90, 255, 255)}
            ],
            "vinyl": [
                {"lower": (0, 0, 0), "upper": (180, 255, 60)}
            ],
            "wood": [
                {"lower": (10, 50, 50), "upper": (30, 255, 255)}
            ]
        }
        self.visual_colors = {
            "plastic": (255, 0, 0),
            "vinyl": (0, 0, 255),
            "wood": (0, 255, 255)
        }

    def _encode_image(self, img):
        retval, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode('utf-8')

    def process(self, image_path):
        img = cv2.imread(image_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # ✅ Edge Detection
        edges = cv2.Canny(gray, 100, 200)
        edges3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # ✅ Color Masking
        masks = {}
        for material, ranges in self.color_ranges.items():
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for r in ranges:
                mask = cv2.inRange(hsv, r["lower"], r["upper"])
                combined_mask = cv2.bitwise_or(combined_mask, mask)

            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            masks[material] = combined_mask

        # ✅ Final Overlay
        overlay = img.copy()
        for material, mask in masks.items():
            color = self.visual_colors[material]
            colored_mask = np.zeros_like(img)
            colored_mask[:] = color
            overlay = cv2.addWeighted(
                overlay, 1, cv2.bitwise_and(colored_mask, colored_mask, mask=mask), 0.5, 0
            )

        # ✅ Legend 추가
        overlay = self.add_legend(overlay)

        # ✅ Base64 변환
        steps = {
            "grayscale": self._encode_image(gray3),
            "hsv": self._encode_image(hsv),
            "mask_plastic": self._encode_image(cv2.cvtColor(masks["plastic"], cv2.COLOR_GRAY2BGR)),
            "mask_vinyl": self._encode_image(cv2.cvtColor(masks["vinyl"], cv2.COLOR_GRAY2BGR)),
            "mask_wood": self._encode_image(cv2.cvtColor(masks["wood"], cv2.COLOR_GRAY2BGR)),
            "edges": self._encode_image(edges3)
        }

        final_result = self._encode_image(overlay)

        return steps, final_result

    def add_legend(self, image):
        legend_height = 50
        legend = np.zeros((legend_height, image.shape[1], 3), dtype=np.uint8)
        x_offset = 10
        for material, color in self.visual_colors.items():
            cv2.rectangle(legend, (x_offset, 10), (x_offset + 30, 40), color, -1)
            label = material.upper()
            cv2.putText(legend, label, (x_offset + 40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            x_offset += 150
        return np.vstack((image, legend))