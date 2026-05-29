# Part 01: LLM & ChatGPT API 가이드북

> **학습 대상**: ChatGPT API를 처음 다루는 개발자  
> **목표**: OpenAI API의 기본 구조부터 실무에서 바로 쓸 수 있는 패턴까지 마스터하기  
> **프로젝트 테마**: AI CCTV 보안 분석 시스템

---

## 1. Part 01 개요

이 파트는 **"LLM이 어떻게 동작하는가"**와 **"ChatGPT API를 어떻게 제어하는가"**를 직접 체험하는 입문 과정입니다.

전체 실습은 **CCTV 보안 상황을 AI가 분석**하는 시나리오를 기반으로 진행되며, 단순한 채팅이 아닌 **실무에 바로 적용 가능한 구조화된 출력**을 만드는 데 초점을 맞춥니다.

### 학습 로드맵

| 순서 | 파일명 | 주요 학습 내용 | 난이도 |
|------|--------|----------------|--------|
| 1 | `ch02_message_struct.py` | messages 구조와 role 이해 | ★☆☆ |
| 2 | `ch02_dotenv_apicall.py` | 환경변수 관리 + 첫 API 호출 | ★★☆ |
| 3 | `ch02_system_prompt_comparison.py` | System Prompt의 위력 | ★★☆ |
| 4 | `ch02_temperature_comparison.py` | temperature 파라미터 | ★★☆ |
| 5 | `ch02_maxtoken_comparison.py` | max_tokens와 비용/품질 트레이드오프 | ★★☆ |
| 6 | `ch02_jsonResponse_parsing.py` | JSON 구조화 출력 (실무 핵심) | ★★★ |
| 7 | `ch02_multiTurnChat.py` | 멀티턴 대화 직접 관리 | ★★★ |

---

## 2. 사전 준비

### 필수 패키지
```bash
pip install python-dotenv openai
```

### .env 파일 구성 (필수)
```env
OPEN_API_KEY=sk-proj-...
OPEN_AI_MODEL=gpt-4o
```

> **보안 주의**: `.env` 파일은 절대 Git에 커밋하지 마세요. `.gitignore`에 반드시 추가해야 합니다.

---

## 3. 파일별 상세 가이드

### 3.1 ch02_message_struct.py — 메시지 구조의 기초

**목적**: API 호출 없이 `messages` 리스트의 구조를 정확히 이해한다.

**핵심 개념**:
- `messages`는 **딕셔너리의 리스트**
- 각 딕셔너리는 반드시 `"role"`과 `"content"`를 가져야 함
- 세 가지 역할(Role)
  - `system`: AI의 정체성과 행동 지침 (대화 전체에 영향)
  - `user`: 사용자가 보내는 질문/지시
  - `assistant`: AI가 이전에 했던 답변 (멀티턴에서 중요)

**실무 인사이트**:
> "API는 기억이 없다."  
> 매번 전체 대화 히스토리를 `messages`로 전달해야 합니다. 이것이 LangChain Memory의 출발점입니다.

---

### 3.2 ch02_dotenv_apicall.py — 첫 번째 실제 API 호출

**목적**: `.env` 파일로 키를 안전하게 관리하고, 실제로 Chat Completions API를 호출한다.

**주요 학습 포인트**:
- `load_dotenv()` + `os.getenv()` 패턴
- `OpenAI()` 클라이언트 생성
- `chat.completions.create()` 기본 호출
- `response.choices[0].message.content` 접근법
- `finish_reason`, `usage` (토큰 사용량) 확인
- 간단한 비용 계산

**핵심 코드 패턴**:
```python
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    max_tokens=100
)
```

---

### 3.3 ch02_system_prompt_comparison.py — System Prompt의 위력

**목적**: 같은 질문이라도 System Prompt에 따라 답변의 품질과 형식이 극적으로 달라진다는 것을 체감한다.

**실험 내용**:
- System 없음
- 일반 어시스턴트 ("You are a helpful assistant.")
- CCTV 전문가 (이 강의에서 사용하는 고품질 System)

**결론**:
실무에서는 **반드시 구체적이고 명확한 System Prompt**를 작성해야 합니다. 특히 보안, 의료, 금융 등 도메인 특화 작업에서는 이것이 성패를 좌우합니다.

---

### 3.4 ch02_temperature_comparison.py — 창의성 vs 일관성

**목적**: `temperature` 파라미터가 답변의 무작위성에 미치는 영향을 직접 확인한다.

**실험 결과 요약**:

| temperature | 특성 | 보안 분석 추천 여부 |
|-------------|------|---------------------|
| 0.0 ~ 0.3   | 매우 일관적, 결정적 | **강력 추천** |
| 0.7         | 적당한 다양성 | 상황에 따라 |
| 1.0 이상    | 창의적이지만 불안정 | **비권장** |

**실무 규칙**:
- 위험도 판단, 분류, JSON 출력 → `temperature=0.0 ~ 0.2`
- 창의적 글쓰기, 브레인스토밍 → `0.7 ~ 1.0`

---

### 3.5 ch02_maxtoken_comparison.py — 토큰 제한과 비용 관리

**목적**: `max_tokens`를 너무 작게 주면 답변이 잘리고, 너무 크게 주면 비용이 불필요하게 증가한다는 것을 이해한다.

**핵심 인사이트**:
- `finish_reason == "length"` → 답변이 잘렸다는 신호
- 보안 분석처럼 **간결한 판단**이 필요한 경우에는 오히려 `max_tokens`를 작게 제한하는 것이 좋을 수 있음

**권장 설정** (이 프로젝트 기준):
- 단순 위험도 판단: 100~200
- 상세 분석 리포트: 300~500
- JSON 구조화 출력: 300~400

---

### 3.6 ch02_jsonResponse_parsing.py — 실무에서 가장 중요한 패턴

**목적**: LLM의 답변을 **파이썬에서 바로 활용 가능한 구조**로 받는 방법을 배운다.

**핵심 기술**:
- `response_format={"type": "json_object"}`
- System Prompt에 JSON 스키마를 명확히 명시 (이게 매우 중요!)
- `temperature=0.0`과 함께 사용하면 거의 100% 유효한 JSON이 나옴
- `json.loads()`로 파싱 후 위험도에 따른 자동 분기 처리

**이 파일이 중요한 이유**:
Part 01의 모든 실습 중 **가장 실무에 가까운 패턴**입니다. 이후 RAG, Agent, FastAPI 연동 등에서 계속 반복해서 사용하게 됩니다.

---

### 3.7 ch02_multiTurnChat.py — 멀티턴 대화 직접 관리

**목적**: API가 "기억"이 없다는 사실을 몸으로 이해하고, 대화 히스토리를 직접 관리하는 방법을 배운다.

**핵심 패턴**:
```python
history.append({"role": "user", "content": user_message})
response = client.chat.completions.create(messages=history)
history.append({"role": "assistant", "content": assistant_reply})
```

**중요한 깨달음**:
- 대화가 길어질수록 토큰 비용이 **선형적으로 증가**한다.
- 이 문제를 해결하기 위해 Part 02에서는 **LangChain Memory**를 사용하게 됩니다.

---

## 4. 핵심 개념 요약표

| 개념 | 설명 | 실무 추천값 | 파일 |
|------|------|-------------|------|
| `system` role | AI의 정체성과 규칙 | 도메인 특화로 작성 | message_struct, system_prompt_comparison |
| `temperature` | 답변의 무작위성 | 보안 분석: 0.0~0.3 | temperature_comparison |
| `max_tokens` | 최대 출력 길이 제한 | 목적에 따라 100~500 | maxtoken_comparison |
| `response_format=json_object` | JSON 강제 출력 | 거의 필수 | jsonResponse_parsing |
| `finish_reason` | 응답 종료 사유 | "length" 주의 | dotenv_apicall, maxtoken |
| `usage` | 토큰 소비량 | 비용 모니터링 필수 | 모든 파일 |

---

## 5. 실습 권장 순서

1. `ch02_message_struct.py` → 구조 이해 (API 호출 없음)
2. `ch02_dotenv_apicall.py` → 첫 호출 성공
3. `ch02_system_prompt_comparison.py` → System의 중요성 체감
4. `ch02_temperature_comparison.py` + `ch02_maxtoken_comparison.py` → 파라미터 실험
5. `ch02_jsonResponse_parsing.py` → **실무 패턴 마스터** (여기서 멈추고 충분히 실습)
6. `ch02_multiTurnChat.py` → 멀티턴의 불편함을 느끼기 (Part 02 연결)

---

## 6. 다음 단계로의 연결

이 파트에서 배운 내용은 **Part 02 LangChain**에서 다음과 같이 발전합니다:

- 수동으로 관리하던 `messages` → LangChain의 `ChatPromptTemplate` + `MessagesPlaceholder`
- 직접 작성하던 멀티턴 로직 → `ConversationBufferMemory`, `ConversationSummaryMemory`
- 수동 JSON 파싱 → `PydanticOutputParser`, `JsonOutputParser`
- 하드코딩된 System Prompt → LangChain의 `SystemMessagePromptTemplate`

---

**작성일**: 2025년 기준  
**대상 프로젝트**: AI CCTV Platform 학습 자료  
**다음 가이드**: `part02_langchain_Guide_Book.md` (예정)

이 가이드북을 따라 실습하면, 단순한 "ChatGPT로 채팅하기" 수준을 넘어 **실제 서비스에 적용 가능한 LLM API 활용 능력**을 갖추게 됩니다.