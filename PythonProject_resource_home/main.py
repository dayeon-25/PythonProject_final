from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil, os, base64
from uuid import uuid4

from models.yolov8 import YOLOWrapper           # plastic, vinyl, wood, count, orig_img
from models.openCV import OpenCVWrapper         # opencv_pro, opencv_result
from models.pca import PCAWrapper               # pca


# ==============================================   Setting   ==========================================================
# 설치 패키지 (시간 오래 걸림)
# pip install fastapi uvicorn ultralytics scikit-learn opencv-python numpy python-multipart
# pip install torch==2.7.0+cpu torchvision==0.22.0+cpu torchaudio==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu

# torch 설치 확인
# python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# --- 출력 결과 ---
# 2.7.0+cpu
# False

# 패키지 설치 리스트 확인
# pip list

# 서버 키기
# uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 서버 끄기
# Ctrl + C / Ctrl + Shift + Esc

# 포트 충돌 시
# netstat -ano | findstr :8000
# TASKKILL /PID 13564 /F

# 서버 실행 확인
# http://localhost:8000/docs
# =====================================================================================================================



app = FastAPI()

# yolo = YOLOWrapper("../weights/best.pt")
# pca = PCAWrapper("../test/models/pca_model.pkl")
# opencv = OpenCVWrapper()

@app.post("/image/analyze")
async def predict(file: UploadFile = File(...)):
    print(f"[LOG] 요청 들어옴: 파일명={file.filename}")


    # # 1️⃣ 업로드 파일 저장
    # os.makedirs("temp", exist_ok=True)
    # save_path = f"temp/{file.filename}"
    # with open(save_path, "wb") as buffer:
    #     shutil.copyfileobj(file.file, buffer)
    # ✅ 요청마다 고유한 파일명 생성
    unique_id = uuid4().hex
    os.makedirs("temp", exist_ok=True)
    save_path = f"temp/{unique_id}_{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    yolo = YOLOWrapper("../weights/best.pt")
    pca = PCAWrapper("../test/models/pca_model.pkl")
    opencv = OpenCVWrapper()

    try:
        # 4️⃣ OpenCV 결과
        #opencv_data = opencv.detect_all_in_one(save_path)
        # opencv_data = {
        #    "plastic": "...base64...",
        #    "vinyl": "...base64...",
        #    "wood": "...base64..."
        # }

        #save_path = save_temp_file(file)
        yolo_result = yolo.predict(save_path)
        pca_result = pca.transform_and_plot(save_path)
        opencv_steps, opencv_final = opencv.process(save_path)

        # # 테스트용 임시 데이터
        # image_bytes = await file.read()
        # encoded_img = base64.b64encode(image_bytes).decode('utf-8')

        # ✅ 파일을 다시 읽기 위해 열린 file 객체를 새로 읽지 말고 저장된 파일에서 읽기
        with open(save_path, "rb") as f:
            encoded_img = base64.b64encode(f.read()).decode('utf-8')

        # print("확인")
        # print(type(yolo_result.get("vinyl", 0.0)))
        # print(yolo_result["rcnn_result"])
        # print(yolo_result.get("rcnn_result"))


        # ✅ DTO 형식 매핑
        response_data = {
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

        print("[LOG] 모든 분석 완료, DTO 구조로 반환")
        return JSONResponse(content=response_data)

    except Exception as e:
        print("[ERROR] 분석 중 오류:", e)
        return JSONResponse(content={"status":1, "error": str(e)}, status_code=500)
