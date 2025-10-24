# 현하님 버전(Form 형태)

from fastapi import FastAPI, Form
import base64
import os
import shutil
from uuid import uuid4

from models.yolov8 import YOLOWrapper           # plastic, vinyl, wood, count, orig_img
from models.openCV import OpenCVWrapper         # opencv_pro, opencv_result
from models.pca import PCAWrapper               # pca

app = FastAPI()

yolo = YOLOWrapper("../weights/best.pt")
pca = PCAWrapper("../test/models/pca_model.pkl")
opencv = OpenCVWrapper()

@app.post("/image/analyze")
async def predict(orig_img: str = Form(...)):

    try:
        print("[LOG] Base64 이미지 수신 완료")

        # 1️⃣ Base64 디코딩
        try:
            img_bytes = base64.b64decode(orig_img)
        except Exception as decode_error:
            print("[ERROR] Base64 디코딩 실패:", decode_error)
            return {"status": 1, "error": "Invalid Base64 String"}

        # # 2️⃣ 디코딩된 이미지를 temp 폴더에 저장
        # os.makedirs("temp", exist_ok=True)
        # save_path = "temp/received_image.jpg"
        # ✅ 요청마다 고유한 파일 저장
        unique_id = uuid4().hex
        os.makedirs("temp", exist_ok=True)
        save_path = f"temp/{unique_id}_received.jpg"


        with open(save_path, "wb") as f:
            f.write(img_bytes)

        print(f"[LOG] 이미지 저장 완료: {save_path}")

        yolo_result = yolo.predict(save_path)
        pca_result = pca.transform_and_plot(save_path)
        opencv_steps, opencv_final = opencv.process(save_path)

        print("비닐 결과 확인")
        print(yolo_result.get("vinyl", 0.0))

        # 4️⃣ 결과 반환
        return {
            "status": 0,
            "orig_img": yolo_result.get("orig_img"),
            "plastic": yolo_result.get("plastic", 0.0),
            "vinyl": yolo_result.get("vinyl", 1.1),
            "wood": yolo_result.get("wood", 0.0),
            "count": yolo_result.get("count", 0),
            "rcnn_result": yolo_result["rcnn_result"],
            "opencv_pro": yolo_result["rcnn_result"],
            "opencv_result": yolo_result["rcnn_result"],
            "pca": yolo_result["rcnn_result"]
        }

    except Exception as e:
        print("[ERROR] 분석 중 오류:", e)
        return {"status": 1, "error": str(e)}