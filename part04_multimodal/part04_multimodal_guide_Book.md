# Part 04: Multimodal (음성 + 비전) 가이드북

> **학습 대상**: Part 01~03을 통해 LangChain과 RAG 기본기를 다진 사람  
> **목표**: **음성(STT, 화자 분리, TTS)**과 **이미지/비전(Vision)**을 LangChain 생태계와 결합하는 실전 능력 기르기  
> **프로젝트 테마**: AI CCTV 보안 관제 시스템 (음성 무전 + 영상 분석)

---

## 1. Part 04 개요

Part 04는 **멀티모달(Multimodal)** 능력을 다루는 파트입니다.

주요 두 축:
- **Audio Track**: Whisper(음성 인식) + pyannote(화자 분리) + TTS(음성 합성)
- **Vision Track**: GPT-4o Vision을 이용한 이미지 분석

이 파트의 핵심은 **"단순히 API를 호출하는 것"을 넘어, 실제 보안 현장에서 사용 가능한 수준의 파이프라인**을 만드는 것입니다.

---

## 2. 폴더 내 주요 파일 분류

### 수업 메인 파일 (Lesson Core)

| 파일명 | 주요 내용 | 난이도 | 비고 |
|--------|-----------|--------|------|
| `ch03_01_make_wav.py` | 기본 WAV 파일 생성 실습 | ★☆☆ | 음성 처리 입문 |
| `ch03_02_whisper_local.py` | 로컬 Whisper 모델 로딩 및 transcribe | ★★☆ | 핵심 |
| `ch03_03_batch_transcribe.py` | 여러 음성 파일 배치 처리 | ★★☆ | 실무 패턴 |
| `ch03_04_multi_dialization.py` | **Whisper + pyannote 화자 분리** (메인 레퍼런스) | ★★★ | **가장 중요** |
| `ch04_01_visionModel_basic.py` | GPT-4o Vision 기본 사용법 | ★★☆ | 비전 입문 |
| `ch04_02_vision_api_call.py` | Vision API 실전 호출 | ★★☆ | 이미지 분석 |

### TTS 관련 파일 (확장/자습)

- `ch03_tts_01_basic.py`
- `ch03_tts_02_speedComp.py`
- `ch03_tts_streaming.py`

> 위 TTS 파일들은 **수업 외 자습용**으로 분류되었습니다.

---

## 3. Chapter별 상세 가이드

### Audio Track (음성 처리)

#### ch03_01_make_wav.py
- 기본적인 WAV 파일을 코드로 생성하는 방법 학습
- 음성 처리 파이프라인의 가장 기초 단계

#### ch03_02_whisper_local.py (중요)
- `openai-whisper` 라이브러리를 사용한 **로컬 Whisper 실행**
- 모델 로딩 전략 (`base`, `small`, `medium` 등)
- `fp16`, `language`, `initial_prompt`, `task` 등 주요 파라미터 실습
- Windows 환경에서의 ffmpeg 문제 대응 방법

#### ch03_03_batch_transcribe.py
- 여러 음성 파일을 효율적으로 처리하는 배치 패턴
- 모델을 한 번만 로드하고 여러 파일에 재사용하는 최적화

#### ch03_04_multi_dialization.py (가장 중요)
**이 파일이 Part 04 Audio Track의 하이라이트입니다.**

주요 학습 내용:
1. Whisper로 음성을 텍스트 + 타임스탬프로 변환
2. pyannote.audio로 **화자 분리(Speaker Diarization)**
3. 두 결과를 결합하여 "누가 언제 말했는지" 구조화
4. `get_speaker_at()`, `merge_consecutive_segments()` 등의 후처리 로직

이 파일은 실제 보안 관제에서 "누가 말했는지"가 중요한 무전/음성 분석 시나리오를 다룹니다.

---

### Vision Track (이미지/비전 분석)

#### ch04_01_visionModel_basic.py
- GPT-4o Vision 모델 기본 사용법
- 이미지 파일을 base64로 인코딩하여 전달하는 방법
- 간단한 이미지 설명 요청 예제

#### ch04_02_vision_api_call.py
- 실제 CCTV/보안 이미지 분석 시나리오 적용
- 구조화된 출력(JSON) 요청
- 비전 모델의 실무 활용 패턴

---

## 4. 핵심 기술 스택

| 영역 | 사용 기술 | 비고 |
|------|-----------|------|
| 음성 인식 | `openai-whisper` (로컬) | `base` 모델 위주로 실습 |
| 화자 분리 | `pyannote.audio` (3.1) | Hugging Face 토큰 필요 |
| 음성 합성 | OpenAI TTS / gTTS | `ch03_tts_*` 시리즈 |
| 비전 분석 | `gpt-4o` Vision | 이미지 → 구조화된 분석 |
| 벡터 DB | ChromaDB | 음성 텍스트 RAG 연동용 |

---

## 5. 추천 학습 순서

### Audio Track (권장 순서)
1. `ch03_01_make_wav.py` — 음성 파일 기초 이해
2. `ch03_02_whisper_local.py` — Whisper 로컬 실행 마스터
3. `ch03_03_batch_transcribe.py` — 배치 처리 패턴
4. `ch03_04_multi_dialization.py` — **화자 분리 전체 파이프라인** (핵심)
5. TTS 파일들 (`ch03_tts_*`) — 선택적으로 자습

### Vision Track
1. `ch04_01_visionModel_basic.py`
2. `ch04_02_vision_api_call.py`

---

## 6. Part 04의 위치와 의미

- **Part 01**: OpenAI API 기초
- **Part 02**: LangChain 구조화
- **Part 03**: RAG (검색 기반 생성)
- **Part 04**: **Multimodal** (음성 + 비전)

Part 04는 지금까지 배운 LangChain 지식을 **음성과 이미지**라는 새로운 모달리티에 적용하는 단계입니다. 특히 `ch03_04_multi_dialization.py`는 실제 보안 관제에서 바로 응용 가능한 수준의 복합 파이프라인을 다룹니다.

---

**작성일**: 2025년 기준  
**대상 프로젝트**: AI CCTV Platform 학습 자료

이 가이드북은 현재 `part04_multimodal` 폴더에 남아 있는 **주요 수업 파일**들을 중심으로 정리했습니다. 자습용 TTS 파일들은 별도로 분류하여 학습 부담을 줄였습니다.