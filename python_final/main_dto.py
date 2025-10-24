from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil, os


# ✅ 외부 모듈 (각 함수는 dict 또는 값으로 결과를 리턴한다고 가정)
from models.yolov8_class import YOLOWrapper      # plastic, vinyl, wood, count, orig_img
from models.openCV import OpenCVWrapper      # opencv_pro, opencv_result
from models.pca_class import PCAWrapper        # pca (가정)


# https://yscho03.tistory.com/327 비동기 처리 참고


app = FastAPI()

yolo = YOLOWrapper("../weights/best.pt")
pca = PCAWrapper("../test/models/pca_model.pkl")
opencv = OpenCVWrapper()


@app.post("/image/analyze")
async def predict(file: UploadFile = File(...)):
    print(f"[LOG] 요청 들어옴: 파일명={file.filename}")


    # 1️⃣ 업로드 파일 저장
    os.makedirs("temp", exist_ok=True)
    save_path = f"temp/{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

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


        # ✅ DTO 형식 매핑
        response_data = {
            "status": 0,
            "orig_img": yolo_result.predict("orig_img"),
            "plastic": yolo_result.get("plastic", 0.0),
            "vinyl": yolo_result.get("vinyl", 0.0),
            "wood": yolo_result.get("wood", 0.0),
            "count": yolo_result.get("count", 0),
            "rcnn_result": yolo_result["rcnn_result"],
            "opencv_pro": opencv_steps,
            "opencv_result": opencv_final,
            "pca": pca_result
        }

        print("[LOG] 모든 분석 완료, DTO 구조로 반환")
        return JSONResponse(content=response_data)

    except Exception as e:
        print("[ERROR] 분석 중 오류:", e)
        return JSONResponse(content={"status":1, "error": str(e)}, status_code=500)