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

        # Legend 추가
        overlay = self.add_legend(overlay)

        # 대비가 심한 불순물 검출 결과로 전송 예정 (현재는 나무로 전송)
        mask_wood = self._encode_image(cv2.cvtColor(masks["wood"], cv2.COLOR_GRAY2BGR))
        #final_result = self._encode_image(overlay)
        final_result = self._encode_image(cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY))

        print("\n[OpenCV] 색상 검출 완료")

        return final_result

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


# 실행 테스트용
if __name__ == "__main__":
    image_path = "C:/Users/301/Desktop/create_image/image_4.png"

    opencv = OpenCVWrapper()
    final_result = opencv.process(image_path)  # ✅ 튜플 언패킹

    # ✅ Base64 → OpenCV 이미지로 변환
    decoded = base64.b64decode(final_result)
    np_img = np.frombuffer(decoded, np.uint8)
    final_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    cv2.imshow("Final Overlay with Legend", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



