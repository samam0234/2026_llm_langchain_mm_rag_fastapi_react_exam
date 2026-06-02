# 사전 훈련 모델로 객체 인식 기본기 익히기
from ultralytics import YOLO

# YOLO ("yolov8n.pt")는 이미 사전 학습된 YOLOv8 nano 모델입니다.
model = YOLO("yolov8n.pt")

# model.predict() : 이미지에 대해서 객체 인식을 실행하는 함수
# source : 소스 (탐지할 이미지 소스)
# conf : 신뢰도 기준. (주어진 값 미만인 클래스는 버림)
# save : true일때 탐지 결과가 그려진 이미지를 자동 저장해라
# 반환값 : list[Result] -> 여러개의 이미지를 분석 시킬 수 있기 때문에
#                         결과 또한 여러개의 Result가 나올 수 있으므로

# results = model.predict(source="./images/man_bus.jpg", conf=0.1, save=True)

source = ["./images/man_bus.jpg", "./images/Blue_parrot.jpeg", "./images/cat.jpeg", "./images/Golden_Retriever.jpeg"]
results = model.predict(source=source, conf=0.1, save=True)

# results는 리스트 타입이다.
# result = results[0]

print(f"yolov8n.pt 모델이 탐지할 수 있는 클래스 : {results[0].names}")
print("=" * 50)

for i, r in enumerate(results, start=1):
    if len(r.boxes) > 0:
        print(f"{i}번째 이미지의 탐지 갯수 : {len(r.boxes)}")
        box = r.boxes[0]
        cls_id_0 = int(box.cls[0])
        print(f"{i}번째 탐지 0번째 클래스번호 : {cls_id_0} => 이름 : {r.names[cls_id_0]}, 신뢰도 {round(float(box.conf[0]), 1)}")
        print(f"픽셀 좌표 {[int(v) for v in box.xyxy[0].tolist()]}")
        print(f"정규화좌표 {[round(v, 3) for v in box.xyxy[0].tolist()]}")
        
        import cv2
        annotated = r.plot()
        
        cv2.imwrite(f"r_{i}.jpg", annotated)
    else:
        print(f"{i}번째 이미지에서는 객체 탐지가 되지 않았습니다.")



# ------------------------------------------------------------------------------

# result.names는 클래스 번호와 클래스 이름을 연결해주는 딕셔너리입니다.
#
# 예를 들어 다음과 같은 형태입니다.
# {
#     0: "person",
#     1: "bicycle",
#     2: "car",
#     ...
# }
#
# YOLO의 내부 결과는 클래스 이름이 아니라 클래스 번호로 들어 있습니다.
# 따라서 번호를 사람이 읽을 수 있는 이름으로 바꾸려면 result.names가 필요합니다.


# result.boxes는 탐지된 박스들의 모음입니다.
# 여기서는 첫 번째 탐지 결과만 꺼내 봅니다.
#
# 주의:
# 만약 탐지된 객체가 하나도 없다면 result.boxes[0]에서 오류가 납니다.
# 실제 서비스에서는 len(result.boxes) > 0인지 먼저 확인하는 것이 안전합니다.


# box.cls는 탐지된 객체의 클래스 번호입니다.
# 예: 0이면 person, 2이면 car
#
# box.cls는 PyTorch Tensor 형태입니다.
# 그래서 box.cls[0]으로 값을 꺼내고 int()로 일반 정수로 바꿉니다.


# box.conf는 confidence, 즉 신뢰도입니다.
# 이 값도 Tensor 형태이므로 box.conf[0]으로 값을 꺼내고 float()로 변환합니다.
#
# round(..., 2)는 소수 둘째 자리까지 반올림합니다.


# box.xyxy는 박스 좌표입니다.
#
# xyxy 형식은 다음 순서를 의미합니다.
# [왼쪽 위 x, 왼쪽 위 y, 오른쪽 아래 x, 오른쪽 아래 y]
#
# box.xyxy[0]은 첫 번째 박스의 좌표입니다.
# .tolist()는 Tensor를 파이썬 리스트로 바꿉니다.
# int(v)는 좌표를 정수로 바꿉니다.


# box.xywhn은 훈련 라벨에서 사용하는 정규화 좌표입니다.
#
# xywhn의 의미:
# x: 박스 중심의 x좌표
# y: 박스 중심의 y좌표
# w: 박스 너비
# h: 박스 높이
# n: normalized, 즉 0~1 사이 비율로 정규화되었다는 뜻
#
# 이 좌표는 화면에 박스를 그릴 때보다
# YOLO 모델을 훈련시킬 때 라벨 파일에서 더 많이 사용합니다.


# result.plot()은 탐지 결과를 이미지 위에 그려줍니다.
#
# 원본 이미지 위에 다음 정보가 표시됩니다.
# - 객체 박스
# - 클래스 이름
# - confidence
#
# 반환값은 numpy 배열입니다.
# 즉, OpenCV에서 사용하는 이미지 형식과 같습니다.