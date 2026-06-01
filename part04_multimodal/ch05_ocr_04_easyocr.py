import easyocr


reader = easyocr.Reader(['ko','en'], gpu=False) # this needs to run only once to load the model into memory
result = reader.readtext('./images/capture3.jpg')

for (bbox, text, confidence) in result:
    # bbox       : 글자를 감싼 네모의 네 꼭짓점 좌표 [좌상, 우상, 우하, 좌하]
    # text       : 인식된 글자
    # confidence : 얼마나 확신하는지 (0~1)
    print(f"인식: {text}  (신뢰도 {confidence:.2f})")