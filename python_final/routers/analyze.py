from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import base64
from models.yolov8_model import YOLOWrapper
from models.pca_class import PCAWrapper
from utils.color_detection import detect_colors

router = APIRouter()

class AnalysisResult(BaseModel):
    yolo_image: str       # Base64 이미지
    yolo_detected: list
    yolo_confidence: list
    pca_graph: str        # Base64 이미지
    pca_values: list
    color_image: str      # Base64 이미지
    color_counts: dict

yolo_model = YOLOWrapper("best.pt")
pca_model = PCAWrapper("pca_model.pkl")

def cv2_to_base64(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode()

@router.post("/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...)):
    # 1. 이미지 읽기
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 2. YOLO 분석
    detected, confidence = yolo_model.predict(img)

    # YOLO 결과 이미지 생성 (예: 박스 그리기)
    yolo_img = img.copy()
    for idx, cls_id in enumerate(detected):
        cv2.rectangle(yolo_img, (10, 10 + idx*30), (110, 40 + idx*30), (0,255,0), 2)
        cv2.putText(yolo_img, f"ID:{cls_id} Conf:{confidence[idx]:.2f}",
                    (15, 30 + idx*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # 3. PCA 분석
    pca_result = pca_model.transform(img)

    # PCA 그래프 그리기
    plt.figure()
    plt.plot(pca_result)
    plt.title("PCA Result")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    pca_graph_base64 = base64.b64encode(buf.read()).decode()
    plt.close()

    # 4. 색상 분석
    color_counts = detect_colors(img)

    # 색상 강조 이미지 생성 (예: 빨간색만 마스크)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red, upper_red = np.array([0,50,50]), np.array([10,255,255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    color_img = cv2.bitwise_and(img, img, mask=mask)

    # 5. 결과 반환
    return AnalysisResult(
        yolo_image=cv2_to_base64(yolo_img),
        yolo_detected=detected,
        yolo_confidence=confidence,
        pca_graph=pca_graph_base64,
        pca_values=pca_result,
        color_image=cv2_to_base64(color_img),
        color_counts=color_counts
    )


# 스프링 부트에서 이미지 받을 때
# byte[] decoded = Base64.getDecoder().decode(base64String);
# BufferedImage img = ImageIO.read(new ByteArrayInputStream(decoded));