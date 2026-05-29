# Part 03: RAG & Vector DB 가이드북

> **학습 대상**: Part 02에서 LCEL과 기본 LangChain 구조를 이해한 사람  
> **목표**: **Retrieval-Augmented Generation (RAG)**의 전체 흐름을 이해하고, 실제 데이터로 검색 기반 답변 시스템을 구축하는 능력 기르기  
> **프로젝트 테마**: CCTV 탐지 로그를 기반으로 한 지능형 보안 분석 시스템

---

## 1. Part 03 개요

Part 01에서는 OpenAI API를 직접 호출했습니다.  
Part 02에서는 LangChain으로 구조를 정리했습니다.

**Part 03에서는 "LLM이 모르는 최신/내부 정보"를 어떻게 주입할 것인가**라는 핵심 문제를 해결합니다.

### 왜 RAG가 필요한가?

- LLM은 학습 시점 이후의 데이터는 알지 못합니다.
- CCTV 탐지 로그처럼 **실시간으로 쌓이는 내부 데이터**를 매번 프롬프트에 넣을 수 없습니다 (토큰 비용 + 컨텍스트 길이 제한).
- **RAG**는 "필요한 정보만 검색해서 가져와서" LLM에게 제공하는 가장 실용적인 해결책입니다.

---

## 2. 학습 로드맵

| Chapter | 주요 주제 | 핵심 키워드 |
|---------|-----------|-------------|
| **ch02** | 임베딩 & Vector DB 기초 | Cosine Similarity, OpenAI Embedding, ChromaDB CRUD, 검색 |
| **ch03** | 실전 RAG 파이프라인 | Document Loader, Text Splitter, Retriever, LCEL RAG Chain, Multi-Query Retriever |

---

## 3. Chapter별 상세 가이드

### Chapter 02: 임베딩과 Vector Database의 이해

이 챕터는 **RAG의 기반 기술**을 직접 체험하는 단계입니다.

#### 주요 파일

- **ch02_01_cosine_similarity.py**
  - RAG의 가장 근본적인 원리인 **코사인 유사도**를 직접 계산하는 예제
  - OpenAI `text-embedding-3-small`으로 텍스트를 벡터로 변환
  - 두 벡터 간의 유사도를 수학적으로 계산하고 시각화
  - **"검색 = 벡터 유사도 계산"** 이라는 핵심 개념을 가장 직관적으로 보여줌

- **ch02_02_chromadb_search.py**
  - 실제 **Vector Database (ChromaDB)** 사용법
  - `OpenAIEmbeddingFunction`을 이용한 자동 임베딩
  - Collection 생성, 문서 추가, 유사도 검색(`query`)
  - 메타데이터 필터링의 중요성

- **ch02_03_chromadb_crud.py**
  - ChromaDB의 기본 CRUD (Create, Read, Update, Delete) 작업 실습
  - 영구 저장 (`PersistentClient`) 방법
  - 실제 운영에서 필요한 데이터 관리 패턴 학습

**이 장에서 얻어야 할 깨달음**:
> "RAG는 단순히 '검색해서 붙여넣기'가 아니라, **의미 기반 검색(Semantic Search)**을 통해 관련 문서를 찾아내는 기술이다."

---

### Chapter 03: 실전 RAG 파이프라인 구축

이 챕터부터가 **진짜 RAG**입니다. LangChain의 LCEL을 활용해 완전한 RAG 시스템을 만듭니다.

#### 주요 파일

- **ch03_01_rag_pipeline.py** (가장 중요)
  - 전체 RAG 파이프라인의 **표준 구조**를 가장 잘 보여주는 파일
  - 단계별 흐름:
    1. **CSVLoader**로 데이터 로드 (`detection_logs.csv`)
    2. **CharacterTextSplitter**로 문서 분할
    3. **OpenAIEmbeddings** + **Chroma**로 벡터 저장
    4. **Retriever** 생성
    5. **PromptTemplate** + LLM + OutputParser로 답변 생성
  - LCEL을 이용한 깔끔한 `Runnable` 체인 구성 (`retriever | format_docs | prompt | llm | parser`)

- **ch03_02_multi_query_retriever.py**
  - 기본 Retriever의 한계 (하나의 쿼리로만 검색 → 누락 발생)
  - **MultiQueryRetriever**의 동작 원리
    - 원본 쿼리 → LLM이 여러 개의 다른 표현으로 재작성
    - 여러 쿼리로 동시에 검색 → 더 넓고 정확한 문서 확보
  - 보안 도메인에서 특히 유용한 고급 검색 기법

**이 장의 핵심 포인트**:
- RAG = **Retrieval** + **Augmented Generation**
- Retriever의 품질이 RAG 전체 성능을 결정한다.
- Multi-Query, Self-Query, Parent-Document Retriever 등 고급 검색 기법의 필요성 이해

---

## 4. 핵심 개념 정리

| 개념 | 설명 | 주요 파일 |
|------|------|-----------|
| **Embedding** | 텍스트를 고차원 벡터로 변환 | ch02_01, ch03_01 |
| **Cosine Similarity** | 두 벡터의 의미적 유사도 측정 | ch02_01 |
| **Vector Database** | 임베딩을 저장하고 유사도 검색을 빠르게 수행 | ch02_02, ch02_03 |
| **Document Loader** | CSV, PDF, TXT 등 외부 데이터를 Document 객체로 변환 | ch03_01 |
| **Text Splitter** | 긴 문서를 의미 단위 청크로 분할 | ch03_01 |
| **Retriever** | 질문에 가장 관련 있는 문서를 찾아주는 컴포넌트 | ch03_01, ch03_02 |
| **LCEL RAG Chain** | Retriever → Prompt → LLM → Parser를 파이프라인으로 연결 | ch03_01 |
| **Multi-Query Retriever** | 하나의 질문을 여러 표현으로 확장하여 검색 품질 향상 | ch03_02 |

---

## 5. 추천 학습 순서

1. **ch02_01_cosine_similarity.py**  
   → RAG의 수학적 기반(코사인 유사도)을 직접 체험

2. **ch02_02_chromadb_search.py**  
   → Vector DB의 기본 동작 이해

3. **ch02_03_chromadb_crud.py**  
   → 실제 운영에서 필요한 데이터 관리 방법 학습

4. **ch03_01_rag_pipeline.py** (가장 중요)  
   → 전체 RAG 파이프라인을 LCEL로 구현. 이 파일을 가장 많이 분석하고 수정해보세요.

5. **ch03_02_multi_query_retriever.py**  
   → Retriever 품질을 높이는 고급 기법 학습

---

## 6. Part 03을 마치고 나면

이 파트를 완료하면 다음과 같은 실력을 갖추게 됩니다:

- 임베딩과 벡터 검색의 원리를 이해
- ChromaDB를 활용해 데이터를 저장하고 검색할 수 있음
- LangChain LCEL로 **완전한 RAG 파이프라인**을 설계할 수 있음
- 단순한 키워드 검색이 아닌 **의미 기반 검색(Semantic Search)** 시스템 구축 가능
- Multi-Query 등 고급 검색 기법을 적용할 수 있는 기반

---

## 7. 다음 단계와의 연결

- **Part 02 (LangChain)**에서 배운 LCEL, PromptTemplate, OutputParser가 RAG에서 어떻게 활용되는지 체감
- **Part 04 (Multimodal)**에서는 음성/이미지 데이터를 RAG에 결합하는 확장된 형태를 다룰 예정
- 실제 서비스에서는 ChromaDB 대신 **Pinecone, Weaviate, PGVector** 등 상용 Vector DB로 전환하는 연습이 필요

---

**작성일**: 2025년 기준  
**대상 프로젝트**: AI CCTV Platform 학습 자료  
**이전 가이드**: `part02_langchain_Guide_Book.md`  
**다음 가이드**: `part04_multimodal_Guide_Book.md` (예정)

RAG는 현재 가장 실용적이고 강력한 LLM 응용 패턴입니다. 이 가이드북을 따라 차근차근 실습하면, 단순한 "ChatGPT에게 물어보기"를 넘어 **자신만의 지식 기반을 가진 지능형 시스템**을 만들 수 있는 역량을 갖추게 될 것입니다.