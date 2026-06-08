# requirments.txt 패키지 분류 정리 문서

> **주의**: 파일 이름이 `requirments.txt` (오타)로 되어 있습니다. 일반적인 이름은 `requirements.txt`입니다.

이 문서는 프로젝트에서 사용하는 `requirments.txt`의 의존성을 **프로젝트의 실제 사용 목적(Part별 모델/기능)** 에 따라 분류하여 정리한 것입니다.

## 1. LangChain & LLM 생태계 (Core AI Logic)

이 프로젝트의 대부분의 지능형 로직을 담당하는 핵심 스택입니다.

| 패키지                        | 용도                                      | 관련 Part     |
|-------------------------------|-------------------------------------------|---------------|
| `langchain`                   | LCEL 파이프라인, Runnable, Agent 등 기본  | Part01, 02    |
| `langchain-core`              | LCEL의 기본 구성 요소                     | All           |
| `langchain-community`         | 커뮤니티 통합 (Chroma 등)                 | Part03        |
| `langchain-openai`            | OpenAI Chat / Embedding 통합              | All           |
| `langchain-chroma`            | ChromaDB와의 LangChain 통합               | Part03        |
| `langchain-text-splitters`    | 문서 분할 (RAG용)                         | Part03        |
| `langgraph`                   | State 기반 Agent / Multi-step 워크플로우  | Part02        |
| `langgraph-checkpoint`        | LangGraph 상태 저장                       | Part02        |
| `langsmith`                   | LLM 호출 추적 및 평가                     | All (선택)    |
| `openai`                      | OpenAI API 직접 호출 (GPT-4o, Whisper 등) | All           |

## 2. RAG & Vector Database

검색 증강 생성(Retrieval-Augmented Generation)을 위한 스택.

| 패키지                  | 용도                              | 관련 Part |
|-------------------------|-----------------------------------|-----------|
| `chromadb`              | 로컬 벡터 데이터베이스            | Part03    |
| `huggingface_hub`       | 임베딩 모델 다운로드 및 관리      | Part03    |
| `sentence-transformers` | (간접 의존) 문장 임베딩           | Part03    |

## 3. 오디오 처리 (STT + 화자 분리 + TTS)

Part04의 핵심 멀티모달 기능.

| 패키지                    | 용도                                      | 관련 Part     |
|---------------------------|-------------------------------------------|---------------|
| `openai-whisper`          | 로컬 Whisper STT 모델                   | Part04        |
| `pyannote-audio`          | 화자 분리 (Speaker Diarization)         | Part04        |
| `pyannote-core`           | pyannote 내부 핵심 라이브러리           | Part04        |
| `torchaudio`              | PyTorch 기반 오디오 처리                | Part04        |
| `soundfile`               | wav 파일 읽기/쓰기                      | Part04        |
| `gTTS`                    | Google TTS (텍스트 → 음성)              | Part04        |
| `torch`                   | PyTorch (Whisper + pyannote 백엔드)     | Part04        |

## 4. 컴퓨터 비전 (Vision)

이미지 분석 관련.

| 패키지             | 용도                          | 관련 Part |
|--------------------|-------------------------------|-----------|
| `opencv-python`    | 이미지 읽기, 전처리, 표시     | Part04    |
| `pytesseract`      | OCR (이미지 → 텍스트)         | Part04    |
| `pillow`           | 이미지 처리 (PIL)             | Part04    |

## 5. PyTorch 딥러닝 백엔드

거의 모든 AI 모델의 기반 엔진.

| 패키지                        | 용도                                      |
|-------------------------------|-------------------------------------------|
| `torch`                       | 메인 딥러닝 프레임워크                    |
| `torchaudio`                  | 오디오 전용 PyTorch                       |
| `torchmetrics`                | 모델 평가 메트릭                          |
| `pytorch-lightning` / `lightning` | 학습 루프 간소화 (일부 사용)          |
| `einops`                      | 텐서 연산 편의 라이브러리                 |
| `safetensors`                 | 안전한 모델 가중치 저장 형식              |

## 6. 데이터 처리 및 과학 계산

| 패키지           | 용도                     |
|------------------|--------------------------|
| `numpy`          | 수치 계산의 기본         |
| `pandas`         | 데이터프레임 처리        |
| `scipy`          | 과학 계산                |
| `scikit-learn`   | 머신러닝 유틸리티        |
| `matplotlib`     | 시각화                   |

## 7. Web / API / 서버

| 패키지          | 용도                          |
|-----------------|-------------------------------|
| `uvicorn`       | ASGI 서버 (FastAPI용)         |
| `pydantic`      | 데이터 검증 및 설정           |
| `python-dotenv` | `.env` 파일 로딩              |

## 8. 관측성 (Observability) - 대부분 Transitive

ChromaDB나 LangSmith가 내부적으로 사용하는 패키지들입니다. 직접 import할 일은 거의 없습니다.

- `opentelemetry-*` 시리즈
- `kubernetes`
- `grpcio`

## 9. 기타 / 개발 유틸리티

| 패키지                    | 비고                              |
|---------------------------|-----------------------------------|
| `tqdm`                    | 진행률 표시                       |
| `rich`                    | 터미널 출력 강화                  |
| `typer`                   | CLI 도구                          |
| `SQLAlchemy` + `alembic`  | 데이터베이스 (일부 실험용)        |
| `optuna`                  | 하이퍼파라미터 최적화             |

---

## 권장 정리 방향 (실제 사용 시)

현재 `requirments.txt`는 **모든 transitive dependency까지 고정**되어 있어서 매우 길고 유지보수가 어렵습니다.

### 추천 구조

```txt
# requirements.txt (직접 의존성만)
langchain
langchain-openai
langchain-chroma
langgraph
openai
openai-whisper
pyannote-audio
gTTS
opencv-python
pytesseract
python-dotenv
torch
torchaudio
numpy
```

그 후 `pip install -r requirements.txt` 후 `pip freeze > requirements.lock` 로 고정하는 방식이 더 좋습니다.

---

**필요하시면** 위 내용을 바탕으로 실제로 `requirements_categories.md` 파일을 만들거나, 더 세밀하게 Part별로 매핑한 버전도 만들어 드리겠습니다.