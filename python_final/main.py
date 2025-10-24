from routers import analyze
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import mysql.connector
from ultralytics import YOLO

from models import yolov8_model

# 패키지 설치
# pip install fastapi
# pip install "uvicorn[standard]"

# 설치 해야 하는 패키지 총집합
# pip install fastapi uvicorn ultralytics scikit-learn opencv-python numpy

app = FastAPI(title="Image Analysis API")

# 라우터 등록, prefix 설정
app.include_router(analyze.router, prefix="/api")
# prefix : 라우터에 붙는 공통 경로를 설정

@app.get("/")
def read_root():
    return {"Hello": "World"}


# ✅ MySQL 연결 설정
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="0000",
        database="final_db",
        charset="utf8mb4"
    )

@app.post("/predict")
def yolo_predict(image_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # 1️⃣ DB에서 이미지 정보 가져오기
        cursor.execute("SELECT id, image_path FROM images WHERE id = %s", (image_id,))
        image = cursor.fetchone()

        if not image:
            raise HTTPException(status_code=404, detail="Image not found")

        image_path = image["image_path"]

        # 2️⃣ YOLO 모델 실행
        results = yolov8_model.YOLOWrapper.predict(image_path)
        result_data = results[0].tojson()  # 결과를 JSON 문자열로 변환

        # 3️⃣ 결과를 DB에 저장
        cursor.execute(
            "INSERT INTO results (image_id, result_json) VALUES (%s, %s)",
            (image_id, result_data)
        )
        conn.commit()
        result_id = cursor.lastrowid  # 저장된 결과의 ID

        # 4️⃣ 결과 ID를 반환 → Spring Boot가 이걸 받음
        return JSONResponse({"result_id": result_id})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()










# 터미널 실행
# uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 파일 구조
# fastapi_project/
# ├─ main.py                # FastAPI 앱 실행, 라우터 등록
# ├─ routers/
#   └─ analyze.py           # /analyze API
# ├─ models/
#   ├─ yolov8_model.py      # YOLOv8 학습/추론 코드
#   └─ pca_class.py         # PCA 분석 코드
# └─ utils/
#   └─ color_detection.py   # OpenCV 색상 검출 함수






# 보내야 하는 데이터
# FastAPI → Spring Boot 전달 방법
# < YOLO >
# - 결과 이미지(박스 그린 이미지)
# - 각 객체별 정확도(confidence)
# - 각 객체별 검출량
# 1. 결과 이미지: OpenCV → JPEG/PNG로 변환 → Base64 인코딩
# 2. 객체별 정확도/검출량: JSON
#
# < PCA >
# - PCA 결과 그래프
# 1. matplotlib로 그래프 그리기 → PNG → Base64 인코딩
# 2. 추가 수치 결과 JSON
#
# < 색상 검출 >
# - 색상별 이미지 강조(옵션)
# - 색상 픽셀 수
# 1. 강조 이미지: OpenCV → PNG → Base64
# 2. 픽셀 수: JSON