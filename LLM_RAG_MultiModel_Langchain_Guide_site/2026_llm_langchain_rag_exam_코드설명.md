# 2026_llm_langchain_mm_rag_fastapi_react_exam 레포지토리 코드 상세 분석

## 현재 코드의 대한 기본 개요

이 GitHub 레포지토리는 **BiNCS AI·데이터 엔지니어 양성 과정**에서 배우는 **LLM, 랭체인, RAG, 멀티모달** 관련 실습 및 시험 대비용 예제 코드 모음입니다.

### 이 코드가 무엇을 하는가?
- **주제**: AI CCTV 보안 분석 시스템 구축
  - OpenCV(또는 YOLO)로 탐지된 객체 정보(사람, 차량 등)를 LLM에게 전달
  - LLM이 위험도(정상/주의/위험)를 판단하고 조치사항을 제안
  - 대화 이력을 기억하는 챗봇, 과거 로그를 검색(RAG)하는 기능까지 점진적으로 확장
  - Part05에서는 YOLOv8을 이용해 **더 정확한 객체 검출 모델을 직접 학습**하고, 수동 라벨링부터 학습까지의 풀 파이프라인을 구축 (기존 OpenCV 탐지보다 고도화된 영상인식 단계)

### 어떤 기능을 위해 존재하는가?
- BiNCS 교육 과정의 **(BiNCS)LLM**, **(BiNCS)랭체인**, **(BiNCS)RAG**, **(BiNCS)자연어처리**, **(BiNCS)프로프트 엔지니어링**, **(BiNCS)OpenCV**, **(BiNCS)영상인식** (YOLO 포함) 능력 단위를 직접 실습하고 이해하기 위한 **단계별 학습 자료**입니다.
- 단순한 API 호출부터 LCEL 파이프라인, 메모리, 에이전트, 벡터DB 기반 RAG까지 **실무에서 바로 쓸 수 있는 LLM 애플리케이션 패턴**을 익히는 것이 목적입니다.
- 최종적으로 FastAPI + React 웹 서비스로 확장할 수 있는 기반 로직을 제공 (repo 이름에 fastapi_react가 들어간 이유).

이 코드는 **"Strawberry Doctor"** 같은 다른 프로젝트와 동일한 맥락에서 **컴퓨터 비전(OpenCV) + LLM**을 결합한 실전형 예제입니다.

---

## 코드 안에 있는 주요 변수명 정리

### 공통으로 자주 등장하는 변수 (여러 파일에서 재사용)
| 변수명 | 타입 | 설명 | 등장 파일 예시 |
|--------|------|------|----------------|
| `api_key` | str | `.env`에서 읽은 OpenAI API 키 | part01, part02 전체 |
| `model` | str | 사용할 모델명 (보통 "gpt-4o") | part01 ch02_*.py |
| `client` | OpenAI | OpenAI 공식 클라이언트 객체 | part01 ch02_dotenv_apicall.py |
| `llm` | ChatOpenAI | LangChain에서 사용하는 OpenAI 채팅 모델 래퍼 | part02 전체 |
| `response` | ChatCompletion / AIMessage | LLM이 반환한 전체 응답 객체 | part01, part02 |
| `answer` | str | LLM이 생성한 실제 텍스트 답변 | part01 |
| `frame_data` | dict | OpenCV 탐지 결과 시뮬레이션 데이터 | part02_lcel, memory_chatbot |
| `detections` | list[dict] | 탐지된 객체 목록 (`class`, `bbox`, `confidence`) | 거의 모든 파일 |
| `result` | dict | 최종 분석 결과 (`risk_level`, `reason`, `action`) | lcel_pipeline.py 등 |
| `history` | InMemoryChatMessageHistory | 대화 이력을 저장하는 메모리 객체 | ch03_real_memory_chatbot.py |
| `cctv_prompt` / `prompt` | ChatPromptTemplate | 시스템+사용자 메시지 템플릿 | lcel_pipeline.py |
| `analysis_chain` | Runnable | `formatter \| prompt \| llm \| parser` 형태의 LCEL 체인 | lcel_pipeline.py |
| `formatter` | RunnableLambda | OpenCV 결과를 프롬프트용 텍스트로 변환하는 함수 | lcel_pipeline.py |
| `json_parser` | JsonOutputParser | LLM 출력을 JSON dict로 자동 변환 | lcel_pipeline.py |
| `SYSTEM_PROMPT` | str | LLM에게 부여하는 역할(페르소나) + 출력 형식 규칙 | memory_chatbot.py |
| `frame_cache` | dict | 분석된 프레임 결과를 저장하는 캐시 | memory_chatbot.py |
| `model` (YOLO) | ultralytics.YOLO | 사전학습 YOLOv8 모델 객체 (yolov8n.pt 등) | ch01_yolo_basic.py, ch03_yolo_train_val_models.py |
| `results` | list[ultralytics.engine.results.Results] | predict() 또는 train() 결과 리스트 | ch01_yolo_basic.py |
| `boxes` (r.boxes) | ultralytics.engine.results.Boxes | 탐지된 박스 정보 (cls, conf, xyxy, xywhn) | ch01_yolo_basic.py |
| `box.cls / box.conf / box.xyxy / box.xywhn` | Tensor | 클래스 ID, 신뢰도, 픽셀 좌표(xyxy), 정규화 라벨 좌표(xywhn) | ch01_yolo_basic.py, labeling scripts |
| `red_boxes_to_yolo_labels` | function | 포토샵 빨간 박스 이미지 → YOLO 라벨 자동 변환 | ch02_yolo_if_not_part.py, ch02_yolo_xywhn_label.py |
| `check_labels` | function | YOLO 라벨 파일 유효성 검사 (5개 값, 클래스 범위, 0~1 정규화) | ch03_yolo_check_labels.py |

### 파일별 특이 변수
- **ch02_dotenv_apicall.py**: `masked_key`, `finish_reason`, `usage`, `input_cost`, `output_cost`, `total_cost`
- **ch03_real_memory_chatbot.py**: `user_msg`, `ai_response`, `_build_messages()`, `analyze_frame()`, `ask()`
- **part03_rag**: `vectorstore` (Chroma), `retriever`, `multi_query_retriever`, `detection_logs.csv` (과거 탐지 로그 데이터)

---

## 코드의 흐름 및 핵심 정리

### 전체 아키텍처 흐름 (단계별 진화)
```
Part01 (직접 API 호출)
   ↓
Part02 (LangChain + LCEL + Memory + Agent)
   ↓
Part03 (RAG + Vector DB)
   ↓
Part04 (Multimodal - 이미지 직접 분석 + OCR)
   ↓
Part05 (YOLO - 객체 검출 사전학습 모델 + 커스텀 라벨링 + 학습)
   ↓
(미래) FastAPI 백엔드 + React 프론트엔드 (YOLO 검출 결과 → LLM 분석)
```

### Part별 핵심 흐름

**1. Part01_llm_chatgpt_api**
- `.env` 파일에서 API 키 안전하게 로드 (`load_dotenv`)
- `client.chat.completions.create()` 직접 호출
- 메시지 구조 (`system` + `user`) 이해
- `temperature`, `max_tokens` 파라미터 영향 실험
- 응답 파싱 (`choices[0].message.content`)
- 토큰 사용량 및 비용 계산 (실무 필수)

**2. Part02_langchain**
- `RunnableLambda`로 커스텀 함수(OpenCV 결과 포맷팅)를 체인에 삽입
- `ChatPromptTemplate` + `|` 연산자로 LCEL 파이프라인 구성
- `JsonOutputParser`로 구조화된 출력 강제 (파싱 오류 감소)
- `InMemoryChatMessageHistory`로 대화 맥락 유지 (메모리)
- `ReAct` 에이전트 + Tool 데코레이터로 도구 사용 패턴 학습

**3. Part03_rag_vectordb**
- `detection_logs.csv`를 벡터화하여 ChromaDB에 저장
- 코사인 유사도 검색으로 관련 과거 로그 검색
- 검색 결과(retrieved documents)를 프롬프트에 추가 (RAG)
- `MultiQueryRetriever`로 질문 변형을 통한 검색 정확도 향상

**4. Part04_multimodal**
- GPT-4o Vision 등 멀티모달 모델 사용
- 실제 CCTV 이미지 프레임을 텍스트 설명 없이 직접 분석
- OCR (pytesseract, EasyOCR)로 이미지 내 텍스트(시간, 위치 등) 추출
- 전처리 (ch05_ocr_02_preprocess_ocr.py)로 OCR 정확도 향상

**5. Part05_yolo**
- ultralytics YOLOv8 사전학습 모델 (yolov8n.pt) 로드 및 추론 (ch01_yolo_basic.py)
- `results[0].names`, `.boxes`, `box.cls/conf/xyxy/xywhn` 구조 이해 (픽셀 좌표 vs YOLO 학습용 정규화 좌표)
- `result.plot()` 으로 시각화된 결과 이미지 생성/저장
- 수동 라벨링 (포토샵 빨간 박스) → 자동 YOLO xywhn 라벨 변환 (ch02_yolo_if_not_part.py, ch02_yolo_xywhn_label.py)
  - OpenCV red masking + contour + boundingRect + 정규화
  - `is_red_rectangle_border` 로 속이 빈 빨간 테두리만 필터링 (노이즈 제거)
- 라벨 품질 자동 검사 (ch03_yolo_check_labels.py): 5개 값, 클래스 범위, 0~1 정규화 검증
- 커스텀 데이터셋으로 YOLO 학습 (ch03_yolo_train_val_models.py)
  - data.yaml (train/val 경로, nc, names)
  - model.train(epochs, imgsz, batch, data=..., project=..., name=...)
  - 결과: runs_cctv/exp1/weights/best.pt (가장 좋은 가중치)
- 목적: 기본 OpenCV 탐지보다 정확한 객체 검출 + 직접 학습 데이터 제작 파이프라인 구축 (CCTV 보안에 특화된 YOLO 모델 제작)

### 가장 중요한 설계 패턴
1. **시스템 프롬프트에 출력 형식 강제** → JSON만 출력하게 하여 후처리 용이
2. **LCEL 파이프라인** → 각 단계를 독립적으로 테스트/교체 가능
3. **메모리 + RAG** → 단일 턴이 아닌 **상태를 가진** 지능형 시스템 구축
4. **OpenCV 결과 → 텍스트 변환** → CV와 LLM을 자연스럽게 연결
5. **YOLO 라벨 포맷 (xywhn 정규화)** → center_x, center_y, width, height 를 이미지 크기로 0~1 정규화. 학습 시 필수.
6. **수동 라벨링 → 자동 변환 파이프라인** → 포토샵 빨간 박스 → OpenCV red detection + contour → YOLO 라벨 (실무 데이터 제작 핵심)
7. **라벨 검증 자동화** → 학습 전 check_labels로 형식 오류 미리 차단 (class 범위, 좌표 0~1, 5개 값)
8. **전이 학습 (YOLOv8n.pt base)** → 사전학습 모델 위에 커스텀 데이터로 fine-tuning. 적은 데이터로도 좋은 성능.

---

## 출력 결과에 대한 상세 설명 및 연관성 분석

### 예시 1: ch02_dotenv_apicall.py 실행 결과
```
API 키 로드 완료: sk-proj-abc...1234
   클라이언트 타입: <class 'openai.OpenAI'>
=== API 첫 번째 호출 성공! ===

GPT 답변 : 위험도: 주의
판단 이유: 새벽 2시에 창고 출입구에서 사람 2명이 탐지되었습니다. ...
종료 사유 : stop

토큰 사용량:
   입력 (prompt)    :   187 토큰
   출력 (completion):    72 토큰
   합계 (total)     :   259 토큰

이번 호출 비용: $0.000XXX  (약 X.XXXX원)
```

**왜 이렇게 나왔는가?**
- `system` 프롬프트에 "항상 한국어로, 간결하게", "위험도는 정상|주의|위험 중 하나"라고 명시했기 때문
- `temperature` 기본값 또는 낮은 값으로 일관된 판단
- `max_tokens=100`으로 답변 길이 제한
- `finish_reason=stop` → 정상적으로 토큰이 다 차지 않고 완료됨
- 비용 계산은 실제 OpenAI 요금 정책을 코드로 재현한 것 (교육 목적)

### 예시 2: ch02_lcel_pipeline.py 실행 결과
```
반환 타입: <class 'dict'>

분석 결과:
  위험도: 주의
  판단 이유: 주차장 A구역에서 사람 2명과 차량 1대가 새벽 시간대에 탐지되었습니다. ...
  조치사항: ...
```

**왜 이렇게 나왔는가?**
- `RunnableLambda(format_detections)`가 OpenCV dict를 프롬프트에 넣기 좋은 텍스트로 변환
- `JsonOutputParser`가 LLM이 출력한 JSON 문자열을 자동으로 Python dict로 변환
- 시스템 프롬프트에 `{{"risk_level": ..., "reason": ..., "action": ...}}` 형식 강제
- `temperature=0`으로 같은 입력에 대해 거의 동일한 JSON 구조 보장

### 예시 3: ch03_real_memory_chatbot.py 실행 결과 (다중 턴)
```
[API 호출] 프레임 1 분석 중... 완료
[API 호출] 프레임 2 분석 중... 완료

운영자: "이전에 탐지된 사람 중에 수상한 행동을 한 사람이 있나요?"
AI: "네, frame_id 1에서 탐지된 두 명의 사람은 ... 이전 분석 결과에 따르면 주의 단계였습니다."
```

**왜 이렇게 나왔는가?**
- `InMemoryChatMessageHistory`에 이전 `HumanMessage` + `AIMessage` 쌍이 모두 저장됨
- `_build_messages()`가 항상 `SystemMessage` + 전체 history + 현재 질문을 LLM에게 전달
- 따라서 LLM이 "이전 분석 결과"를 기억하고 자연스럽게 답변할 수 있음
- 메모리가 없으면 매번 독립적인 답변만 하고 맥락을 잃음

### RAG (Part03) 출력 특징
- `retriever.invoke("주차장 새벽 인원 증가")` → `detection_logs.csv`에서 유사한 과거 로그 검색
- 검색된 문서(context)를 프롬프트에 추가 → LLM이 "과거에도 비슷한 사례가 있었고, 당시에는..." 식으로 답변
- **할루시네이션(환각) 감소**와 **사실 기반 답변**이 핵심 효과

### 전체 출력 결과가 가지는 교육적 의미
- **구조화된 출력(JSON)**: 실제 서비스에서 프론트엔드(React)가 파싱하기 쉽도록 설계
- **토큰/비용 표시**: LLM 서비스 운영 시 필수적인 모니터링 항목 교육
- **메모리 + RAG 조합**: 단순 챗봇이 아닌 **지능형 보안 관제 시스템**으로 발전 가능성을 보여줌
- **OpenCV 연동**: BiNCS 과정에서 배운 컴퓨터 비전 내용을 LLM과 자연스럽게 연결하는 실전 예제
- **YOLO 라벨 제작 파이프라인**: 포토샵(또는 다른 툴)으로 수동 주석 → OpenCV 자동 변환 → 검증 → 학습. 실무에서 가장 중요한 "데이터 제작" 능력 직접 체험.

### Part05 YOLO 상세 코드 설명

**ch01_yolo_basic.py**
- `YOLO("yolov8n.pt")` : Ultralytics에서 제공하는 사전학습 COCO 모델 로드 (80 클래스, nano 버전은 가볍고 빠름)
- `model.predict(source=..., conf=0.1, save=True)` : 여러 이미지 동시 추론. conf 이하 박스는 필터링.
- `results[0].names` : {0: 'person', 1: 'bicycle', ...} 클래스 매핑 딕셔너리. YOLO 내부는 클래스 ID(int)만 사용하므로 사람이 읽을 이름으로 매핑 필요.
- `r.boxes` : 탐지된 모든 박스 정보 객체.
  - `box.cls[0]` : 클래스 ID (int)
  - `box.conf[0]` : 신뢰도 (0~1)
  - `box.xyxy[0]` : [x1,y1,x2,y2] 픽셀 좌표 (왼쪽위, 오른쪽아래). 화면에 그리기 좋음.
  - `box.xywhn[0]` : [cx, cy, w, h] 정규화 좌표 (0~1). **YOLO 학습 라벨(.txt) 형식**으로 사용. 중심 좌표 + 크기 비율.
- `r.plot()` : 원본 이미지 위에 박스+라벨+conf를 그린 numpy 배열 반환. cv2.imwrite로 저장 가능.
- 교육 포인트: **xyxy (시각화) vs xywhn (학습 라벨)** 차이 이해. 실제 YOLO 학습 시 라벨은 반드시 xywhn 정규화 포맷이어야 함.

**ch02_yolo_if_not_part.py / ch02_yolo_xywhn_label.py**
- 포토샵으로 이미지에 빨간색 **빈 사각형 테두리**를 그려서 라벨링하는 실전 워크플로우.
- `cv2.inRange(img, lower, upper)` + BGR 마스킹으로 빨간 픽셀만 추출.
- `cv2.morphologyEx(..., MORPH_CLOSE)` : 테두리 선이 끊어진 부분 연결.
- `cv2.findContours(..., RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` + `cv2.boundingRect` 로 외접 박스 (xywh 픽셀).
- `is_red_rectangle_border` : 4변에 충분한 빨간 선분이 있고, 내부는 빨간색이 적어야 함 (빨간 물체 전체가 아니라 "박스 테두리"만 골라냄).
- xyxy(픽셀) → xywhn(정규화) 변환 공식:
  ```
  cx = (x + w/2) / image_width
  cy = (y + h/2) / image_height
  nw = w / image_width
  nh = h / image_height
  ```
- `save_yolo_label` : 이미지 basename + .txt 로 labels/train/ 에 저장 (YOLO 표준 구조).
- 디버그: 초록 박스로 검증 이미지 생성.
- 교육 포인트: **실제 라벨링 비용 절감 + 자동화**. 포토샵으로 빠르게 주석 → 스크립트로 YOLO 포맷 변환. 대량 데이터 제작 시 필수.

**ch03_yolo_check_labels.py**
- `check_labels(label_dir, num_classes)` : 학습 전 라벨 품질 자동 검사.
- 검사 항목:
  1. 한 줄 = 정확히 5개 토큰 (class + 4 coords)
  2. class : 0 <= cls < num_classes
  3. 모든 좌표 : 0.0 <= v <= 1.0 (정규화 위반 검출)
- glob + re로 윈도우 경로 문제 해결.
- 문제 있으면 리스트로 반환 → 학습 전에 미리 "비참한 결과" 방지.
- 교육 포인트: YOLO 학습에서 **라벨 품질이 모델 성능의 80%**를 결정. 자동 검증 스크립트는 실무 필수 도구.

**ch03_yolo_train_val_models.py**
- `YOLO("yolov8n.pt").train(...)` : 전이학습 시작. nano 베이스 모델 위에 우리 CCTV 데이터로 fine-tuning.
- 주요 파라미터:
  - `data="yolo_train/data.yaml"` : train/val 이미지+라벨 경로, nc(클래스 수), names 정의
  - `epochs=40`, `imgsz=320` (작게 해서 빠른 실험), `batch=8`
  - `device="cpu"` (GPU 있으면 0)
  - `project="runs_cctv"`, `name="exp1"` → runs_cctv/exp1/weights/best.pt 생성
- 결과:
  - best.pt : validation mAP가 가장 좋았던 가중치 (실제 추론에 사용)
  - last.pt : 마지막 에폭
- 교육 포인트: **YOLOv8 Ultralytics API의 단순함**. data.yaml 하나만 잘 만들면 train 한 줄로 끝. 실제 CCTV에 특화된 검출 모델을 직접 만들 수 있음.

### Part05에서 새로 등장한 핵심 개념
- **YOLO 라벨 포맷 (class cx cy w h normalized 0~1)**
- **전이 학습 + 커스텀 데이터셋 학습 파이프라인**
- **수동 주석(포토샵) → 자동 라벨 변환 스크립트**
- **학습 전 라벨 검증 자동화**
- **train/val split** 과 data.yaml 구조 (YOLO 표준)
- **best.pt / last.pt** 구분과 실전 사용

이 단계는 "단순히 모델 쓰기"에서 "**내 데이터로 직접 모델을 가르치고 라벨을 제작하는 풀 파이프라인**"으로 넘어가는 중요한 전환점입니다. BiNCS 과정의 영상인식 + 데이터 엔지니어링 역량을 종합적으로 보여줍니다.

---

## BiNCS 교육 과정 능력단위와의 연계

제공된 이미지의 능력단위명과 직접 매핑됩니다:

- **(BiNCS)LLM** + **(BiNCS)자연어처리** → Part01 직접 API 호출 + 프롬프트
- **(BiNCS)랭체인** → Part02 LCEL, Memory, Agent
- **(BiNCS)프로프트 엔지니어링** → 모든 파일의 system prompt 설계
- **(BiNCS)RAG** → Part03 ChromaDB + retriever
- **(BiNCS)OpenCV** + **(BiNCS)영상인식** → frame_data의 detections 구조 (YOLO 결과 연동)
- **(BiNCS)UI 인터페이스** + **(BiNCS)파이썬 서버 제작** → 향후 FastAPI + React 확장 지점
- **(BiNCS)영상인식 + Object Detection** → Part05 YOLO (사전학습 모델 추론, 커스텀 라벨링, data.yaml 기반 학습, best.pt 생성)

이 레포지토리는 단순한 코드 모음이 아니라, **BiNCS 과정에서 배운 내용을 하나로 통합**하여 실무 프로젝트(예: CCTV 보안 대시보드)로 발전시킬 수 있는 **종합 실습 자료**입니다.

---

**작성일**: 2026년 5월 기준 (Part05 YOLO 추가 반영)  
**목적**: BiNCS 수강생이 이 코드를 보고 각 단계의 의도와 출력 결과를 정확히 이해할 수 있도록 정리

이 파일을 통해 코드의 **왜(why)**와 **어떻게(how)**를 명확히 파악하시기 바랍니다.

**Part05 추가 노트**: YOLO 단계는 "LLM이 분석할 좋은 검출 결과"를 만드는 데이터 파이프라인 구축을 목표로 합니다. 단순히 모델을 쓰는 게 아니라, **내 데이터로 라벨을 만들고, 검증하고, 학습시키는 전체 과정**을 경험하는 것이 핵심입니다. (이전 Part들의 OpenCV/YOLO 결과가 이 단계의 입력이 됩니다.)
