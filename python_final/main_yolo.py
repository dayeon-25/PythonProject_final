# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil, os

from models.yolov8 import run_inference  # ✅ 외부 파일에서 모델 사용

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print(f"[LOG] 요청 들어옴: 파일명={file.filename}")

    # 업로드 파일 저장
    os.makedirs("temp", exist_ok=True)
    save_path = f"temp/{file.filename}"
    with open(save_path, "wb") as buffer:



        shutil.copyfileobj(file.file, buffer)

    # ✅ 모델 추론 실행
    try:
        result_data = run_inference(save_path)
        print("[LOG] YOLO 추론 완료")
        return JSONResponse(content=result_data)
    except Exception as e:
        print("[ERROR] Inference failed:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)
