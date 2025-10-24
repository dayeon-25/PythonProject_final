import cv2
import numpy as np


# 각 불순물별 이미지를 띄워 놓고 각 색상값을 사용자한테 입력 받아서 해당 색상값을 입혔을 때 색상 검출이 어떻게 이루어지는지를 보여즘
#




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

        # 색상별 시각화 색 (BGR)
        self.visual_colors = {
            "plastic": (255, 0, 255),   # Magenta
            "vinyl": (0, 0, 255),     # Red
            "wood": (0, 255, 0)     # Green
        }

    def process_and_show(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print("❌ 이미지 파일을 찾을 수 없습니다. 경로를 확인하세요:", image_path)
            return

        # 🔽 일정 크기로 resize
        fixed_width = 700
        fixed_height = 500
        img = cv2.resize(img, (fixed_width, fixed_height))
        #해상도 : 800x600 / 1024x768 / 1280x720 / 1920x1080


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

            # Morphology
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

        overlay_with_legend = self.add_legend(overlay)

        # ✅ 화면에 출력
        cv2.imshow("Original Image", img)
        cv2.imshow("Grayscale", gray)
        cv2.imshow("HSV (visualized as BGR)", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        cv2.imshow("Mask - Plastic", cv2.cvtColor(masks["plastic"], cv2.COLOR_GRAY2BGR))
        cv2.imshow("Mask - Vinyl", cv2.cvtColor(masks["vinyl"], cv2.COLOR_GRAY2BGR))
        cv2.imshow("Mask - Wood", cv2.cvtColor(masks["wood"], cv2.COLOR_GRAY2BGR))
        cv2.imshow("Edges", edges)
        cv2.imshow("Final Overlay with Legend", overlay_with_legend)

        print("✅ 이미지가 표시되었습니다. 창을 닫거나 아무 키나 누르면 프로그램이 종료됩니다.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def add_legend(self, image):
        legend_height = 50
        legend = np.zeros((legend_height, image.shape[1], 3), dtype=np.uint8)

        x_offset = 10
        for material, color in self.visual_colors.items():
            cv2.rectangle(legend, (x_offset, 10), (x_offset + 30, 40), color, -1)
            cv2.putText(legend, material.upper(), (x_offset + 40, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            x_offset += 150

        return np.vstack((image, legend))


# ================================
# ✅ 실행 파트 (이미지 경로만 변경해서 실행)
# ================================
if __name__ == "__main__":
    image_path = "C:/Users/301/Desktop/create_image/image_yolo.png"  # ✅ 여기에 실제 이미지 파일명을 입력
    #image_path = "C:/Users/301/Desktop/ocr.png"
    opencv = OpenCVWrapper()
    opencv.process_and_show(image_path)