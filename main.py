from fastapi import FastAPI, File, UploadFile, APIRouter, Form
from fastapi.responses import JSONResponse
import cv2, base64, numpy as np
from uuid import uuid4
import os

from fastapi import FastAPI
import pandas as pd
from sklearn.linear_model import LinearRegression

from models.yolov8 import YOLOWrapper
#from models.yolo import YOLOWrapper
from models.opencv import OpenCVWrapper
from models.pca import PCAWrapper



# ==============================================   Setting   ==========================================================
# 설치 패키지 (시간 오래 걸림)
# pip install fastapi uvicorn ultralytics scikit-learn opencv-python numpy python-multipart pandas
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

# Router 생성
router = APIRouter()


# =======================================
# 이미지 규격 체크
# =======================================
@router.post("/image/validation")
async def image_check(image: UploadFile = File(...)):
    print(f"\n[IMAGE SIZE] 이미지 확인 요청 들어옴: 파일명 = {image.filename}")

    try:
        # 이미지 바이트 읽기
        image_bytes = await image.read()
        np_img = np.frombuffer(image_bytes, np.uint8)

        # OpenCV로 디코딩
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # 디코딩 실패 시 (지원되지 않는 파일 등)
        if img is None:
            print("[IMAGE SIZE] 이미지 디코딩 실패: 파일이 이미지가 아닐 수 있습니다.")
            return JSONResponse(content={"status": 1}, status_code=400)

        # 해상도 확인
        height, width = img.shape[:2]
        print(f"[IMAGE SIZE] 감지된 해상도: {width}x{height}")

        if width == 1920 and height == 1080:
            print("[IMAGE SIZE] 이미지 규격 통과")
            return JSONResponse(content={"status": 0})
        else:
            print("[IMAGE SIZE] 이미지 규격 불만족")
            return JSONResponse(content={"status": 1})

    except Exception as e:
        print(f"[ERROR] 이미지 처리 중 오류 발생: {e}")
        return JSONResponse(content={"status": 1}, status_code=500)




# =======================================
# 이미지 분석 API (YOLO + PCA + OpenCV)
# =======================================
@router.post("/image/analyze")
async def predict(file: UploadFile = File(...)):
    print(f"\n[LOG] 요청 들어옴: 파일명 = {file.filename}")

    # 요청마다 고유한 파일명 생성
    unique_id = uuid4().hex
    os.makedirs("temp", exist_ok=True)
    save_path = f"temp/{unique_id}_{file.filename}"
    with open(save_path, "wb") as buffer:
        buffer.write(await file.read())

    # ✅이미지 해상도 검사
    img = cv2.imread(save_path)
    if img is None:
        print("[ERROR] 이미지 로딩 실패")
        return JSONResponse(content={"status": 1, "message": "INVALID_IMAGE"}, status_code=400)

    height, width = img.shape[:2]
    print(f"[LOG] 감지된 해상도: {width}x{height}")

    if width != 1920 or height != 1080:
        print("[LOG] 이미지 규격 불일치. 분석 중단.")
        return JSONResponse(content={"status": 1, "message": "INVALID_SIZE"}, status_code=200)

    print("[LOG] 이미지 규격 통과. 분석 시작...")

    yolo = YOLOWrapper("../weights/best.pt")
    pca = PCAWrapper()
    opencv = OpenCVWrapper()

    try:
        yolo_result = yolo.predict(save_path)
        pca_result = pca.analyze(save_path)
        opencv_final = opencv.process(save_path)

        # ✅ DTO 형식 매핑
        response_data = {
            "status": 0,
            "orig_img": yolo_result.get("orig_img"),
            "plastic": yolo_result.get("plastic", 0.0),
            "vinyl": yolo_result.get("vinyl", 0.0),
            "wood": yolo_result.get("wood", 0.0),
            "count": yolo_result.get("count", 0),
            "rcnn_result": yolo_result["rcnn_result"],
            "opencv_result": opencv_final,
            "pca": pca_result
        }

        print("\n[LOG] 모든 분석 완료, DTO 구조로 반환")

        # 보내는 결과 확인용
        print("\n========== 🔍 분석 결과 ==========")
        print(f"[YOLO] 파일명={file.filename}")
        print(f"[YOLO] 총 감지 객체 수: {response_data['count']}")
        print(f"[YOLO] 나무 비율: {response_data['wood']}%")
        print(f"[YOLO] 비닐 비율: {response_data['vinyl']}%")
        print(f"[YOLO] 플라스틱 비율: {response_data['plastic']}%")

        for key, value in response_data.items():
            if value is None:
                print(f"⚠️ {key} 값이 None 입니다!")
            else:
                # 길이가 길 경우, 이미지인 경우는 'None 아님'만 출력
                print(f"✅ {key} 값 정상 반환됨 (None 아님)")
        print("==================================\n")

        return JSONResponse(content=response_data)

    except Exception as e:
        print("\n[ERROR] 분석 중 오류:", e)
        return JSONResponse(content={"status": 1, "error": str(e)}, status_code=500)


# =======================================
# HSV 모핑 API
# =======================================
# 1. RGB 이미지를 입력받아 HSV 이미지로 변환한다.
# 2. 색상의 범위에 따라 특정 색상의 객체를 추출하는 마스크를 생성한다.
# 3. 생성한 마스크에 따라 이미지를 계산하여 특정한 색상의 객체만 추출되는 결과 이미지를 만든다.
# https://bradbury.tistory.com/64 hsv 이미지 참고

def cv2_to_base64(img):
    """OpenCV 이미지를 Base64 문자열로 변환"""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode("utf-8")

@router.post("/morphing")
async def morphing(
    file: UploadFile = File(...),
    h: int = Form(...),
    s: int = Form(...),
    v: int = Form(...)):

    try:
        print(f"[모핑] 요청: {file.filename}, HSV=({h},{s},{v})")

        # 이미지 로드
        image_bytes = await file.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # 1. Grayscale 이미지
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Edge Detection
        edges = cv2.Canny(gray, 100, 200)

        # 3. Sobel Edge
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = cv2.convertScaleAbs(cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0))
        sobel_img = cv2.cvtColor(sobel_mag, cv2.COLOR_GRAY2BGR)

        # HSV 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 사용자가 입력한 값을 기준으로 범위 설정 #### 사용자가 margin 도 입력할 수 있게 수정할 수도 있음;
        h_margin = 10  # hue는 민감하니까 좁게
        sv_margin = 40  # saturation, value는 더 넓게

        # lower = np.array([0, 0, 0])
        # upper = np.array([h, s, v])
        lower = np.array([max(h - h_margin, 0), max(s - sv_margin, 0), max(v - sv_margin, 0)])
        upper = np.array([min(h + h_margin, 179), min(s + sv_margin, 255), min(v + sv_margin, 255)])

        mask = cv2.inRange(hsv, lower, upper)

        # 4. HSV 마스크 흑백
        hsv_gray = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 5. HSV 마스크 컬러
        masked_result = cv2.bitwise_and(img, img, mask=mask)

        return JSONResponse(content={
            "opencv_pro1": cv2_to_base64(gray),
            "opencv_pro2": cv2_to_base64(edges),
            "opencv_pro3": cv2_to_base64(sobel_img),
            "opencv_pro4": cv2_to_base64(hsv_gray),
            "opencv_pro5": cv2_to_base64(masked_result),
        })

    except Exception as e:
        return JSONResponse(content={
            "opencv_pro1": f"error: {str(e)}",
            "opencv_pro2": f"error: {str(e)}",
            "opencv_pro3": f"error: {str(e)}",
            "opencv_pro4": f"error: {str(e)}",
            "opencv_pro5": f"error: {str(e)}",
        })


# =======================================
# 그래프 예측치 전달 API
# =======================================
@router.get("/api/pred")
async def get_predictions() :
# def get_predictions() :
    df = pd.read_csv('한국환경공단_순환골재 폐기물 데이터_20211130.csv', encoding='cp949')

    df_grouped = df.groupby('보고년도')[['판매량_톤']].sum()
    df_grouped = df_grouped.reset_index()

    df_grouped['법적규제'] = np.where(df_grouped['보고년도'] >= 2016, 1, 0)

    X = df_grouped[['보고년도', '법적규제']].values
    y = df_grouped['판매량_톤'].values

    model = LinearRegression()
    model.fit(X, y)

    years_to_predict = np.array([2022, 2023, 2024, 2025]).reshape(-1, 1)
    legal_change = np.array([1, 1, 1, 1]).reshape(-1, 1)
    X_new = np.hstack((years_to_predict, legal_change))

    predictions = model.predict(X_new)

    actual_data = [{"year": int(year), "sales": float(sales)} for year, sales in zip(df_grouped['보고년도'], df_grouped['판매량_톤'])]
    predicted_data = [{"year": int(year), "sales": float(prediction)} for year, prediction in zip(years_to_predict.flatten(), predictions)]
    data = actual_data + predicted_data
    print("[예측값 확인]")
    print(data)
    return data


# FastAPI에 Router 등록
app.include_router(router)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
