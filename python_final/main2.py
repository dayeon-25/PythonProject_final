from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil, os, base64
from ultralytics import YOLO
import cv2

app = FastAPI()
#model = YOLO("best.pt")  # YOLOv8 모델 로드
model = YOLO("C:/Users/301/Desktop/final_result80/runs/yolo_seg_train/weights/best.pt")

@app.post("/image")
async def predict(file: UploadFile = File(...)):
    print(f"[LOG] 요청 들어옴: 파일명={file.filename}")
    # 업로드 파일 저장
    save_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 모델 추론
    print("[LOG] YOLO 추론 시작...")
    results = model(save_path)
    result = results[0]
    print("[LOG] YOLO 추론 완료")


    # 결과 이미지 저장
    result_path = f"temp/result_{file.filename}"
    print(f"[LOG] 결과 이미지 경로: {result_path}")
    result.save(filename=result_path)
    print("[LOG] 결과 이미지 저장 완료")

    # 결과 통계 계산
    names = model.names
    counts = {}
    total = 0

    for box in result.boxes:
        cls = int(box.cls[0])
        name = names[cls]
        counts[name] = counts.get(name, 0) + 1
        total += 1

    ratios = {k: round(v / total * 100, 2) for k, v in counts.items()} if total > 0 else {}

    # 결과 이미지 base64 인코딩
    with open(result_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

        # --- 로그 찍기 ---
    print("Detected Objects:", counts)
    print("Ratios:", ratios)
    print("Types:", type(counts), type(ratios))

    return JSONResponse(content={
        "detected_objects": counts,
        "ratios": ratios,
        "total_detected": total,
        "result_image": encoded_image
    })


# 서버 키기
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
