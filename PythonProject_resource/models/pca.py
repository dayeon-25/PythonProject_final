import os
import numpy as np
import joblib
import cv2
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# 타원 그리기 추가(평균 타원)
from matplotlib.patches import Ellipse

# 타원 그리기 추가(전체 타원)
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

from ultralytics import YOLO


class PCAWrapper:
    def __init__(self):
        # -------------------------------
        # 1) 저장된 PCA 및 데이터 로드
        # -------------------------------
        self.pca = joblib.load("C:/Users/301/PycharmProjects/PythonProject_resource/models/pca/pca_model.pkl")
        self.scaler = joblib.load("C:/Users/301/PycharmProjects/PythonProject_resource/models/pca/scaler.pkl")
        self.explained = np.load("C:/Users/301/PycharmProjects/PythonProject_resource/models/pca/pca_explained.npy")         # PC1/PC2가 설명하는 분산 비율
        self.features_std = np.load("C:/Users/301/PycharmProjects/PythonProject_resource/models/pca/features_std.npy")       # 표준화된 특징(학습 때 사용)
        self.labels = np.load("C:/Users/301/PycharmProjects/PythonProject_resource/models/pca/labels.npy")                   # 각 샘플의 클래스 라벨 문자열

        self.pca_result = self.pca.transform(self.features_std)      # 기준 데이터의 2D 좌표

        # YOLO 모델 로드(사용자 이미지에서 비율 추출용)
        self.model = YOLO("../weights/best.pt")

        # 클래스 인덱스 ↔ 이름 매핑
        self.CLASS_NAMES = {0: 'wood', 1: 'vinyl', 2: 'plastic'}
        self.CLASS_LIST = ['wood', 'vinyl', 'plastic']

        # 산점도 색상 지정
        self.colors = {'wood': 'blue', 'vinyl': 'orange', 'plastic': 'green'}

    # -------------------------------
    # 2) 사용자 이미지 비율 추출 함수
    # -------------------------------
    def extract_ratios(self, image_path):
        """
        YOLO 세그멘테이션으로 사용자 이미지에서
        wood / vinyl / plastic의 '면적 비율(픽셀수/전체픽셀)'을 계산해 [w, v, p]로 반환.
        실패 시 None.
        """
        results = self.model(image_path)

        img = cv2.imread(image_path)
        if img is None:
            print(f"[PCA] 이미지 로드 실패: {image_path}")
            return None, True  # 이미지 자체 로드 실패도 마스크 없음처럼 처리

        h, w, _ = img.shape
        total_area = h * w

        class_areas = {name: 0 for name in self.CLASS_LIST}

        mask_detected = False

        for result in results:
            masks = result.masks
            if masks is None:
                continue
            for mask, cls in zip(masks.data, result.boxes.cls):
                mask_detected = True
                cls_name = self.CLASS_NAMES[int(cls)]
                class_areas[cls_name] += np.sum(mask.cpu().numpy().astype(np.uint8))

        ratios = [
            class_areas['wood'] / total_area,
            class_areas['vinyl'] / total_area,
            class_areas['plastic'] / total_area
        ]
        return ratios, not mask_detected  # mask_detected == False → 마스크 없음임을 True로 반환



    def draw_class_cluster_circle(self, ax):
        """
        각 클래스의 평균 위치를 중심으로 하는 1표준편차 타원(원 형태)을 그림
        """
        for cls in self.CLASS_LIST:
            # 해당 클래스 인덱스 추출
            idx = np.where(self.labels == cls)[0]
            class_points = self.pca_result[idx]

            # PCA 좌표의 평균과 표준편차 계산
            mean_x, mean_y = np.mean(class_points, axis=0)
            std_x, std_y = np.std(class_points, axis=0)

            # 타원(원) 객체 생성
            circle = Ellipse((mean_x, mean_y),
                             width=std_x * 3,  # x축 범위 (표준편차 기반으로 조절 가능)
                             height=std_y * 3,  # y축 범위
                             color=self.colors[cls],
                             alpha=0.2,
                             fill=True,
                             label=f"{cls} range")

            ax.add_patch(circle)

    # ===== 타원 그리는 함수 =====
    def draw_convex_hull(self, ax):
        for cls in self.CLASS_LIST:
            idx = np.where(self.labels == cls)[0]
            class_points = self.pca_result[idx]

            if len(class_points) < 3:
                print(f"[ConvexHull] {cls} 클래스는 점이 3개 미만이라 헐을 만들 수 없음")
                continue

            try:
                hull = ConvexHull(class_points)
                vertices = class_points[hull.vertices]
                polygon = Polygon(vertices, closed=True, alpha=0.25, color=self.colors[cls], label=f"{cls} range")
                ax.add_patch(polygon)
                print(f"[ConvexHull] {cls} 클래스 헐 생성 완료")
            except Exception as e:
                print(f"[ConvexHull] {cls} 클래스 헐 생성 실패: 데이터가 한 직선상에 존재하거나 평면 형성이 불가함 → 타원을 대체 적용")
                # ConvexHull이 실패할 경우, 표준편차 기반 '타원 범위'를 대신 표시
                self.draw_class_cluster_circle(ax)


    def draw_max_range_ellipse(self, ax):
        """
        각 클래스의 PCA 분포 전체 범위를 감싸는 타원을 그림.
        중심은 평균(mean_x, mean_y),
        크기는 평균으로부터 좌우/상하로 퍼져 있는 최대 거리 기반.
        """
        for cls in self.CLASS_LIST:
            idx = np.where(self.labels == cls)[0]
            class_points = self.pca_result[idx]

            if len(class_points) == 0:
                continue

            # 1) 평균(타원의 중심)
            mean_x, mean_y = np.mean(class_points, axis=0)

            # 2) 평균 기준 최대 거리 (전체 직선 범위를 감싸는 방식)
            max_x_dist = np.max(np.abs(class_points[:, 0] - mean_x))
            max_y_dist = np.max(np.abs(class_points[:, 1] - mean_y))

            # 3) 타원 생성 (확장율은 선택적으로 1.1 정도 주면 시각적으로 여유 있게 보임)
            ellipse = Ellipse(
                (mean_x, mean_y),  # 중심 위치
                width=max_x_dist * 2 * 1.1,  # 가로 길이 = 양방향 최대거리 ×2
                height=max_y_dist * 2 * 1.1,  # 세로 길이
                color=self.colors[cls],
                alpha=0.25,
                fill=True,
                label=f"{cls} range"
            )

            ax.add_patch(ellipse)

    def draw_oriented_ellipse(self, ax, scale=2.0, min_minor=0.2):
        """
        공분산의 고유벡터(주성분 방향)로 회전한 타원을 그림.
        - 중심: 각 클래스 평균
        - 장축/단축: sqrt(고유값) * scale
        - 수치 불안정(음수 고유값, 거의 1D 분포) 보정 포함
        """
        for cls in self.CLASS_LIST:
            idx = np.where(self.labels == cls)[0]
            class_points = self.pca_result[idx]
            if len(class_points) < 2:
                continue  # 분산 계산 불가

            # 중심
            mean_x, mean_y = np.mean(class_points, axis=0)
            centered = class_points - np.array([mean_x, mean_y])

            # 공분산 → 고유분해
            cov = np.cov(centered, rowvar=False)
            if not np.all(np.isfinite(cov)):
                continue

            eigvals, eigvecs = np.linalg.eigh(cov)  # 2x2
            # 수치오차로 생긴 음수 제거
            eigvals = np.clip(eigvals, 0.0, None)

            # 큰 고유값 = 주축이 되도록 정렬
            order = eigvals.argsort()[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            # 장축/단축 표준편차
            major = np.sqrt(eigvals[0])
            minor = np.sqrt(eigvals[1]) if eigvals[1] > 0 else 0.0

            # 단축이 0에 가깝다면, 주축에 수직한 벡터로 실제 분산을 재추정
            if minor < min_minor:
                u = eigvecs[:, 0]  # 주축 단위벡터
                v = np.array([-u[1], u[0]])  # 주축에 수직
                proj_minor = centered @ v  # 수직 방향 투영값
                minor = np.std(proj_minor)
                # 최소 두께 보장
                minor = max(minor, min_minor)

            width = 2 * scale * major
            height = 2 * scale * minor

            # 회전 각도 (라디안→도)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

            ell = Ellipse((mean_x, mean_y),
                          width=width, height=height, angle=angle,
                          color=self.colors[cls], alpha=0.25, fill=True,
                          label=f"{cls} range")
            ax.add_patch(ell)


    # -------------------------------
    # 3) 사용자 이미지 분석 및 시각화 함수
    # -------------------------------
    def analyze(self, user_image_path):
        user_ratios, is_no_mask = self.extract_ratios(user_image_path)
        print("\n[PCA] 사용자 이미지 비율:", user_ratios)

        if is_no_mask:
            print("[PCA] 사용자 이미지에서 마스크가 검출되지 않았습니다. (비율 = [0,0,0])")

        user_ratios_std = self.scaler.transform([user_ratios])
        user_pca = self.pca.transform(user_ratios_std)[0]

        print("[PCA] 사용자 이미지 PCA 좌표:", user_pca)

        # -------------------------------
        # 4) PCA 시각화 (기준 데이터 + 사용자 이미지)
        # -------------------------------
        #plt.figure(figsize=(8, 6))
        fig, ax = plt.subplots(figsize=(8, 6))


        for cls in self.CLASS_LIST:
            idx = np.where(self.labels == cls)[0]
            plt.scatter(
                self.pca_result[idx, 0], self.pca_result[idx, 1],
                color=self.colors[cls], label=cls, alpha=0.6
            )

        # 클래스 영역 원으로 감싸기
        #self.draw_class_cluster_circle(ax)
        #self.draw_convex_hull(ax)
        #self.draw_max_range_ellipse(ax)
        self.draw_oriented_ellipse(ax)

        # 사용자 이미지 표시
        if is_no_mask:
            ax.scatter(user_pca[0], user_pca[1], color='gray', marker='*', s=250, label='User Image (No Mask)')
        else:
            ax.scatter(user_pca[0], user_pca[1], color='red', marker='*', s=250, label='User Image')

        ax.set_title("PCA Scatter: Dataset vs User Image")
        ax.set_xlabel(f"PC1 ({self.explained[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({self.explained[1] * 100:.1f}%)")
        ax.legend()
        ax.grid(True)

        # 그래프를 이미지로 변환 후 base64 인코딩
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)  # 메모리 누수 방지

        return img_base64





# -------------------------------
# 테스트 실행
# -------------------------------
if __name__ == "__main__":
    #image_path = "C:/Users/301/Desktop/636962_516882_4909.jpg"
    #image_path = "C:/Users/301/Desktop/create_image/goljae.png"
    #image_path = "C:/Users/301/PycharmProjects/PythonProject_final/image/vinyl/10_X199_C587_0224_2.jpg"
    image_path = "C:/Users/301/Desktop/pca_image/plastic/cocacola_mp4-0_jpg.rf.025a4fad03323fec2d4c56b9ca9d8faf.jpg"
    analyzer = PCAWrapper()
    result = analyzer.analyze(image_path)

    # ✅ Base64 → OpenCV 이미지로 변환
    decoded = base64.b64decode(result)
    np_img = np.frombuffer(decoded, np.uint8)
    final_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    cv2.imshow("PCA Result", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()