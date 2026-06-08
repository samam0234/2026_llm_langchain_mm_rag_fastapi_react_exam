from ultralytics import YOLO

# YOLO("yolov8n.pt") : 사전훈련 모델에서 '시작'합니다.
#   = 전이학습. 기본기 있는 모델 위에 우리 데이터를 얹어 가르칩니다.
model = YOLO("yolov8n.pt")

# model.train(...) : 훈련을 실행하는 함수. 학습이 끝나면 새 모델 파일이 생깁니다.
model.train(
    data="yolo_train/data.yaml",  # ← 우리가 만든 안내문
    epochs=40,                   # 데이터 전체를 40번 반복 학습
    imgsz=320,                   # 학습용 이미지 크기(가로=세로 픽셀)
    batch=8,                     # 한 번에 8장씩 묶어서 학습
    device="cpu",                # GPU가 있으면 device=0 (훨씬 빠름)
    project="runs_cctv",         # 결과를 저장할 폴더 이름
    name="exp1",                 # 이번 실험 이름
)
# 훈련이 끝나면 가중치(학습 결과)가 아래 위치에 저장됩니다:
#   runs_cctv/exp1/weights/best.pt   ← 가장 성적 좋았던 모델 (이걸 씁니다)
#   runs_cctv/exp1/weights/last.pt   ← 마지막 순간의 모델
print("훈련 완료! best.pt 가 생성되었습니다.")