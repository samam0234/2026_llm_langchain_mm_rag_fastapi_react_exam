# Part 02: LangChain 가이드북

> **학습 대상**: Part 01에서 OpenAI API를 직접 다뤄본 사람  
> **목표**: LangChain을 이용해 **유지보수 가능하고, 확장 가능한 LLM 애플리케이션**을 만드는 방법 배우기  
> **프로젝트 테마**: AI CCTV 보안 분석 시스템 (Part 01의 연장)

---

## 1. Part 02 개요

Part 01에서는 **OpenAI API를 직접 호출**하면서 많은 불편함을 경험했습니다:

- 프롬프트를 문자열로 매번 조립해야 함
- JSON 파싱 코드를 반복 작성
- 대화 기록(히스토리)을 직접 관리
- 복잡한 로직(도구 호출, 조건분기)을 if문으로 직접 구현

**LangChain**은 이런 반복적이고 번거로운 작업을 **컴포넌트화**하고, **선언적(Declarative)**으로 연결할 수 있게 해주는 프레임워크입니다.

### 학습 로드맵

| Chapter | 주요 주제 | 핵심 키워드 |
|---------|-----------|-------------|
| **ch01** | LangChain이 왜 필요한가? | Runnable, RunnableLambda, 기본 체인 |
| **ch02** | LCEL (LangChain Expression Language) | PromptTemplate, ChatPromptTemplate, OutputParser, LCEL 파이프라인 (`\|`) |
| **ch03** | Memory (기억) | ConversationBufferMemory, SummaryMemory, 실전 메모리 챗봇 |
| **ch04** | Agent & Tools | @tool 데코레이터, ReAct 패턴, Tool Calling Agent |

---

## 2. Part 01 vs Part 02 비교

| 항목 | Part 01 (순수 OpenAI API) | Part 02 (LangChain) |
|------|---------------------------|---------------------|
| 프롬프트 관리 | f-string으로 직접 조립 | `ChatPromptTemplate`로 재사용 가능하게 관리 |
| 출력 파싱 | `json.loads()` 매번 작성 | `JsonOutputParser`, `PydanticOutputParser` |
| 대화 기록 | `history` 리스트 직접 append | `Memory` 컴포넌트가 자동 관리 |
| 복잡한 로직 | if/else로 직접 구현 | Agent가 LLM 판단으로 Tool 자동 선택 |
| 유지보수성 | 낮음 (코드 중복 많음) | 높음 (컴포넌트 재사용) |

---

## 3. Chapter별 상세 가이드

### Chapter 01: LangChain의 필요성 이해

#### 주요 파일

- **ch01_whyLangchain.py**
  - LangChain 없이 반복 호출할 때의 문제점 3가지를 실제 코드로 보여줌
  - 프롬프트 수정 어려움, 파싱 코드 중복, 재사용 불가
  - **"그래서 LangChain이 필요하다"**는 동기를 부여하는 파일

- **ch01_langchain.py**
  - LangChain의 가장 기본적인 4단계 패턴 소개
    1. PromptTemplate 만들기
    2. LLM (ChatOpenAI) 만들기
    3. OutputParser 만들기
    4. `\|` 연산자로 Chain 연결
  - LCEL의 가장 단순한 예제

- **ch01-1_runnableLambda.py**
  - `RunnableLambda`를 이용한 커스텀 함수 체인 연결 방법
  - LangChain의 `Runnable` 인터페이스 이해

**이 장의 핵심 메시지**:
> "LangChain은 단순히 OpenAI를 편하게 쓰는 라이브러리가 아니라, **복잡한 LLM 애플리케이션을 구조적으로 만들기 위한 설계 도구**다."

---

### Chapter 02: LCEL 마스터하기 (가장 중요)

이 챕터가 Part 02의 **진짜 핵심**입니다.

#### 주요 파일

- **ch02_prompt_template.py**
  - `ChatPromptTemplate.from_messages()` 사용법
  - System + Human 메시지 템플릿 작성
  - 변수 바인딩 (`{frame_id}`, `{detections_text}` 등)

- **ch02_lcel_pipeline.py**
  - LCEL 파이프라인의 전체 흐름을 가장 잘 보여주는 파일
  - `format_detections` (전처리) → Prompt → LLM → JsonOutputParser
  - 실제 CCTV 분석 파이프라인 예제

- **ch02_output_parser.py**
  - `StrOutputParser` vs `JsonOutputParser` 비교
  - 파서의 역할과 장점 설명

- **ch02_format_detections.py**
  - OpenCV 탐지 결과를 LangChain 입력 형식으로 변환하는 전처리 함수
  - Part 03(RAG)에서도 비슷한 전처리 패턴이 자주 등장

**이 장에서 반드시 이해해야 할 것**:
- **LCEL의 `\|` 연산자**가 어떻게 데이터 흐름을 만드는지
- PromptTemplate이 하는 일
- OutputParser가 하는 일
- Runnable 인터페이스의 `invoke()`, `batch()`, `stream()` 메서드

---

### Chapter 03: Memory (기억 관리)

Part 01의 `ch02_multiTurnChat.py`에서 느꼈던 "히스토리를 직접 관리하는 피로감"을 해결하는 장입니다.

#### 주요 파일

- **ch03_memory_compare.py**
  - 기억이 없는 LLM vs 기억이 있는 LLM의 차이를 시뮬레이션으로 명확히 보여줌
  - Memory의 필요성을 가장 잘 설명하는 파일

- **ch03_simple_buffer_memony.py**
  - 가장 기본적인 `ConversationBufferMemory`
  - 모든 대화 내용을 그대로 저장

- **ch03_simple_smmary_memory.py**
  - `ConversationSummaryMemory`
  - 긴 대화를 LLM이 요약해서 저장 (토큰 비용 절감)

- **ch03_real_memory_chatbot.py**
  - 실제 ChatOpenAI + Memory를 결합한 실전 챗봇 예제
  - CCTV 상황을 지속적으로 기억하면서 대화하는 형태

**핵심 포인트**:
- Memory는 단순히 "이전 대화를 저장"하는 것이 아니라, **LLM에게 어떤 맥락을 전달할지** 결정하는 전략이다.
- Buffer vs Summary의 트레이드오프를 이해하는 것이 중요

---

### Chapter 04: Agent & Tools

가장 어렵지만, 가장 강력한 개념입니다.

#### 주요 파일

- **ch04_react_simulate.py**
  - ReAct (Reason + Act) 패턴을 **수동으로 시뮬레이션**
  - LLM이 실제로 어떤 사고 과정을 거치는지 보여줌 (Thought → Action → Observation 루프)

- **ch04_tool_decorator.py**
  - `@tool` 데코레이터를 사용해 일반 Python 함수를 Tool로 변환하는 방법
  - Tool의 docstring이 LLM에게 매우 중요하다는 점 강조

- **ch04_langchain_pipeline.py**
  - 여러 컴포넌트를 연결한 복합 파이프라인 예제

- **ch04_langchain_agent_final.py**
  - **실제 LLM + Tool Calling Agent**의 완성판
  - Mock이 아닌 진짜 `ChatOpenAI` + `bind_tools` 사용
  - LLM이 스스로 어떤 Tool을 쓸지 판단하는 진짜 Agent 동작 확인

**이 장의 목표**:
> "개발자가 if문으로 '이 질문에는 이 Tool을 써'라고 하드코딩하는 것이 아니라, **LLM이 스스로 판단**하게 만드는 것"

---

## 4. 핵심 개념 정리

| 개념 | 설명 | 주요 등장 파일 |
|------|------|----------------|
| **Runnable** | LangChain의 모든 컴포넌트가 따르는 공통 인터페이스 | 전체 |
| **LCEL (`\|`)** | 컴포넌트를 선언적으로 연결하는 표현 언어 | ch02 시리즈 |
| **ChatPromptTemplate** | 재사용 가능한 프롬프트 템플릿 | ch02_prompt_template |
| **OutputParser** | LLM 출력을 구조화된 데이터로 변환 | ch02_output_parser |
| **Memory** | 대화 맥락을 유지하는 전략 | ch03 시리즈 |
| **Tool** | LLM이 호출할 수 있는 외부 기능 | ch04_tool_decorator |
| **Agent** | LLM이 Tool 사용을 스스로 결정하는 구조 | ch04_langchain_agent_final |
| **ReAct** | Reasoning + Acting을 반복하는 Agent 패턴 | ch04_react_simulate |

---

## 5. 추천 학습 순서

1. **ch01_whyLangchain.py** → 왜 LangChain이 필요한지 공감하기
2. **ch01_langchain.py + ch01-1_runnableLambda.py** → 기본 구조 이해
3. **ch02_prompt_template.py** → PromptTemplate 마스터
4. **ch02_lcel_pipeline.py** → LCEL 파이프라인의 전체 흐름 체득 (가장 중요)
5. **ch02_output_parser.py + ch02_format_detections.py**
6. **ch03_memory_compare.py** → Memory의 필요성 이해
7. **ch03 시리즈 나머지** → 실제 Memory 종류 실습
8. **ch04_react_simulate.py** → Agent의 사고 과정 이해
9. **ch04_tool_decorator.py**
10. **ch04_langchain_agent_final.py** → 진짜 Agent 완성

---

## 6. Part 02를 마치고 나면

이 파트를 끝내면 다음과 같은 능력을 갖추게 됩니다:

- LCEL로 깔끔한 파이프라인을 설계할 수 있다
- Memory를 상황에 맞게 선택할 수 있다
- 간단한 Agent + Tool 시스템을 직접 만들 수 있다
- Part 03 (RAG)와 Part 04 (멀티모달)에서 나오는 복잡한 구조를 이해할 수 있는 기반이 생긴다

---

**작성일**: 2025년 기준  
**대상 프로젝트**: AI CCTV Platform 학습 자료  
**이전 가이드**: `part01_llm_chatgpt_api_Guide_Book.md`  
**다음 가이드**: `part03_rag_vectordb_Guide_Book.md` (예정)

LangChain은 "마법의 도구"가 아니라, **LLM 애플리케이션을 체계적으로 만들기 위한 설계 언어**입니다. 이 가이드북을 따라 차근차근 실습하면, 단순한 데모를 넘어 실서비스 수준의 구조를 설계할 수 있는 실력을 갖추게 될 것입니다.