import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.decomposition import PCA

class PCAWrapper:
    def __init__(self, model_path="pca_model.pkl"):
        try:
            self.pca = joblib.load(model_path)
            print(f"[PCA] 모델 로드 완료: {model_path}")
        except FileNotFoundError:
            print("[WARN] PCA 모델이 없습니다. fit()을 먼저 실행하세요.")
            self.pca = None

    def transform_and_plot(self, image_path):
        """
        ✅ 이미지를 PCA로 변환하고, 산점도 그래프 이미지를 base64로 반환
        """
        if self.pca is None:
            return "PCA model not loaded"

        try:
            img = cv2.imread(image_path)
            if img is None:
                return "Invalid image path"

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            flat = gray.flatten().reshape(1, -1)

            transformed = self.pca.transform(flat)[0]  # 1개의 PCA 결과 벡터

            # ✅ 산점도 그리기
            fig, ax = plt.subplots()
            ax.scatter(transformed[0], transformed[1], marker='o')
            ax.set_title("PCA Result Scatter Plot")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

            # ✅ 이미지 → base64 변환
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")

            return img_base64  # ✅ DTO의 pca 필드에 들어갈 값

        except Exception as e:
            return f"PCA transform failed: {str(e)}"