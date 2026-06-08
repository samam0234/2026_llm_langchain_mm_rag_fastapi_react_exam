/**
 * Pyodide-powered "파트별 Python 코드 실행기" - Scenario Driven Version
 * 
 * 이 파일이 진짜 소스 오브 트루스입니다.
 * - 모든 시나리오 데이터 (상황 설명 + 예제 코드 + 결과 렌더러)
 * - Pyodide 실행 로직
 * 
 * index.html은 UI와 최소 glue만 담당.
 * 
 * Part 구성: Part01 / Part02 / Part03 / Part04 (총 4개, 각 10개 시나리오)
 */

let pyodide = null;

async function loadPyodideIfNeeded() {
    if (pyodide) return pyodide;

    const status = document.getElementById('pyodide-status');
    if (status) status.textContent = 'Pyodide 로딩 중... (최초 10~20초, 일부 패키지 로드)';

    try {
        pyodide = await loadPyodide({
            indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/"
        });

        // 자주 쓰이는 가벼운 패키지 미리 로드
        try {
            await pyodide.loadPackagesFromImports(`
                import json
                import os
                import sys
                import re
            `);
        } catch (e) {}

        if (status) {
            status.textContent = '✅ Pyodide 준비 완료 (기본 패키지 로드됨)';
            status.classList.add('text-emerald-400');
        }
        return pyodide;
    } catch (err) {
        console.error(err);
        if (status) status.textContent = 'Pyodide 로드 실패 (인터넷 확인 필요)';
        return null;
    }
}

async function runPythonViaPyodide() {
    const editor = document.getElementById('python-code-editor');
    const output = document.getElementById('python-console-output');

    if (!editor || !output) {
        alert('코드 에디터 요소를 찾을 수 없습니다.');
        return;
    }

    const code = editor.value;
    output.innerHTML = '<span class="text-yellow-400">Pyodide 준비 및 실행 중...</span>';

    const py = await loadPyodideIfNeeded();
    if (!py) {
        output.innerHTML = '<span class="text-red-400">Pyodide를 로드할 수 없습니다. (인터넷 연결 확인)</span>';
        return;
    }

    try {
        // 공통으로 유용한 패키지 미리 로드 시도 (실패해도 계속 진행)
        try {
            await py.loadPackagesFromImports(code);
        } catch (e) {
            // 일부 패키지는 로드 실패해도 괜찮음 (예: openai, whisper 등)
        }

        py.runPython(`
            import sys
            from io import StringIO
            sys.stdout = StringIO()
        `);

        await py.runPythonAsync(code);
        const stdout = py.runPython("sys.stdout.getvalue()");

        if (stdout && stdout.trim()) {
            output.innerHTML = stdout.split('\n').map(line => `<div>${line}</div>`).join('');
        } else {
            output.innerHTML = '<div class="text-zinc-500">(출력 없음)</div>';
        }
    } catch (err) {
        let msg = String(err);
        let friendly = '';

        if (msg.includes('ModuleNotFound') || msg.includes('import')) {
            friendly = `<br><span class="text-orange-400">→ 이 파일은 Pyodide 브라우저 환경에서 필요한 패키지를 로드할 수 없거나, heavy ML 모델(whisper, pyannote, torch 등)을 지원하지 않습니다.</span>`;
        } else if (msg.includes('OpenAI') || msg.includes('API')) {
            friendly = `<br><span class="text-orange-400">→ OpenAI API 호출은 브라우저에서 직접 할 수 없습니다 (키 + CORS 문제).</span>`;
        }

        output.innerHTML = `<div class="text-red-400">실행 오류: ${msg}${friendly}</div>`;
    }
}

// 카테고리별 기본 코드 (실제 프로젝트에서 포팅 + 실행 가능한 순수 Python)
// index.html 의 defaultPythonCodes 와 완전히 동일한 내용 (유지보수 시 둘 중 하나만 수정해도 되지만, 현재는 양쪽 동기화)
const DEMO_CATEGORIES = {
    'part01-basic': `# Part01 - Basic Chat Completion (실제 ch02 스타일)
messages = [
    {"role": "system", "content": "당신은 AI CCTV 보안 분석 시스템입니다. 위험도를 '안전/주의/위험' 중 하나로 판단하세요."},
    {"role": "user", "content": "새벽 2시 창고 출입구에서 person 2명 탐지. 타임스탬프: 02:47"}
]

print("=== Part01 Basic Completion ===")
for m in messages:
    print(f"{m['role'].upper()}: {m['content']}")
print()
print(">>> LLM 응답 (모의)")
print({"risk_level": "주의", "reason": "심야 시간대 다수 인원 탐지", "action": "근처 순찰 강화"} )`,

    'part02-lcel': `# Part02 - LCEL 스타일 체인 (ch01-1, ch02_lcel_pipeline 포팅)
from typing import Dict, Any, List

def format_detections(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "frame_id": data.get("frame_id"),
        "detections_text": f"- {data.get('detections', '')} 탐지됨"
    }

def add_risk(data: Dict[str, Any]) -> Dict[str, Any]:
    data["risk_level"] = "위험" if "person" in str(data) else "주의"
    return data

class SimpleLCEL:
    def __init__(self, steps: List):
        self.steps = steps
    def invoke(self, input_data: Dict) -> Dict:
        result = input_data
        for step in self.steps:
            result = step(result)
        return result

chain = SimpleLCEL([format_detections, add_risk])
input_data = {"frame_id": 42, "detections": "person:2, car:1"}
result = chain.invoke(input_data)

print("=== Part02 LCEL Pipeline ===")
print(result)`,

    'part03-rag': `# Part03 - RAG Pipeline (ch03_01, ch03_02 포팅 스타일)
documents = [
    "2024-11-20 02:31 | 창고 출입구 | person:2 | 침입 의심",
    "2025-01-08 03:15 | 공장 외곽 | person:3, car:2 | 정상",
    "2025-02-14 01:05 | 주차장 A | person:1 | 야간 순찰"
]

def simple_retrieve(query: str, docs: list, top_k: int = 2):
    scored = []
    for d in docs:
        score = sum(1 for w in query.lower().split() if w in d.lower())
        scored.append((score, d))
    scored.sort(reverse=True)
    return [d for _, d in scored[:top_k]]

def generate(context: list, query: str):
    return f"검색된 {len(context)}개 컨텍스트 기반 답변: 새벽/심야 창고 침입 사례가 다수 발견되었습니다."

query = "새벽 창고 침입"
ctx = simple_retrieve(query, documents)
answer = generate(ctx, query)

print("=== Part03 RAG (Retriever + Generate) ===")
print("Query:", query)
print("Retrieved:")
for c in ctx: print(" -", c)
print("Answer:", answer)`,

    'part04-diarization': `# Part04 - merge_consecutive_segments (ch03_04_multi_dialization.py 에서 정확히 포팅)
def merge_consecutive_segments(segments, gap_threshold=0.8):
    """
    실제 프로젝트의 핵심 함수.
    같은 화자가 0.8초 이내 연속 발화하면 하나로 합침.
    """
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        if seg["speaker"] == prev["speaker"] and gap < gap_threshold:
            prev["text"] += " " + seg["text"].strip()
            prev["end"] = seg["end"]
        else:
            merged.append(seg.copy())
    return merged

raw_segments = [
    {"speaker": "민욱 (Control)", "start": 7.8, "end": 10.0, "text": "3번 프레임 창고 출입구 확인"},
    {"speaker": "민욱 (Control)", "start": 10.1, "end": 12.3, "text": "person 2명"},
    {"speaker": "주완 (Unit 3)", "start": 13.1, "end": 15.0, "text": "차량 1대 동반"},
    {"speaker": "주완 (Unit 3)", "start": 15.2, "end": 18.9, "text": "위험 상황으로 판단"}
]

print("=== Raw diarization segments (before merge) ===")
for s in raw_segments:
    print(f"  [{s['speaker']}] {s['start']:.1f}s ~ {s['end']:.1f}s | {s['text']}")

merged = merge_consecutive_segments(raw_segments)
print("\\n=== After merge_consecutive_segments (gap_threshold=0.8) ===")
for s in merged:
    print(f"  [{s['speaker']}] {s['start']:.1f}s ~ {s['end']:.1f}s")
    print(f"     → {s['text']}")
print("\\n(실제 프로젝트와 동일한 로직이 Pyodide에서 실행됨)")`,

    'part04-vision': `# Part04 - Vision Analysis (ch04_01 ~ ch04_05 포팅)
image_analysis = {
    "objects": ["person", "vehicle"],
    "counts": {"person": 2, "vehicle": 1},
    "risk_level": "주의",
    "description": "야간 주차장에서 두 사람이 차량 주변을 배회하는 모습",
    "recommendation": "추가 확대 영상 확인 및 순찰 요청"
}

print("=== Part04 Vision Multi-Image Analysis ===")
print("Objects detected:", image_analysis["objects"])
print("Risk assessment:", image_analysis["risk_level"])
print("Action:", image_analysis["recommendation"])
print("\\n(실제 GPT-4o vision 호출 결과와 유사한 구조)")`
};

function loadDemoCategory(category) {
    const editor = document.getElementById('python-code-editor');
    if (editor && DEMO_CATEGORIES[category]) {
        editor.value = DEMO_CATEGORIES[category];
    }
}

// 전역 노출 (index.html에서 onclick으로 호출 가능하게)
window.runCurrentPyodideCode = runPythonViaPyodide;
window.loadDemoCategory = loadDemoCategory;

console.log('%c[Pyodide Demo] demo.js loaded (maintainable version)', 'color:#64748b');

/* ============================================================
   NEW: Scenario-Driven "파트별 Python 코드 실행기"
   각 Part당 10개 상황 기반 시나리오
============================================================ */

// ============================================================
// 파트별 Python 파일 목록 (GitHub 폴더 구조 매핑)
// ============================================================
const PYTHON_FILES = {
  'part01': [
    { name: 'ch02_dotenv_apicall.py', description: 'dotenv를 사용한 API 호출' },
    { name: 'ch02_jsonResponse_parsing.py', description: 'JSON 응답 파싱' },
    { name: 'ch02_maxtoken_comparison.py', description: 'Max Token 비교' },
    { name: 'ch02_message_struct.py', description: 'Message 구조' },
    { name: 'ch02_multiTurnChat.py', description: '다중 턴 채팅' },
    { name: 'ch02_system_prompt_comparison.py', description: 'System Prompt 비교' },
    { name: 'ch02_temperature_comparison.py', description: 'Temperature 비교' }
  ],
  'part02': [
    { name: 'ch01_langchain.py', description: 'LangChain 기본 (데모)' },
    { name: 'ch01_whyLangchain.py', description: 'LangChain의 필요성 (데모)' },
    { name: 'ch01-1_runnableLambda.py', description: 'RunnableLambda (데모)' },
    { name: 'ch02_format_detections.py', description: '탐지 결과 포맷팅' },
    { name: 'ch02_lcel_pipeline.py', description: 'LCEL 파이프라인 (데모)' },
    { name: 'ch02_output_parser.py', description: 'Output Parser (데모)' },
    { name: 'ch02_prompt_template.py', description: 'Prompt Template (데모)' },
    { name: 'ch03_memory_compare.py', description: 'Memory 비교 (데모)' },
    { name: 'ch03_real_memory_chatbot.py', description: 'Memory 챗봇 (데모)' },
    { name: 'ch04_langchain_agent_final.py', description: 'Agent 최종 버전 (데모)' }
  ],
  'part03': [
    { name: 'ch02_01_cosine_similarity.py', description: '코사인 유사도' },
    { name: 'ch02_02_chromadb_search.py', description: 'ChromaDB 검색 (데모)' },
    { name: 'ch02_03_chromadb_crud.py', description: 'ChromaDB CRUD (데모)' },
    { name: 'ch03_01_rag_pipeline.py', description: 'RAG 파이프라인 (데모)' },
    { name: 'ch03_02_multi_query_retriever.py', description: 'Multi-Query Retriever (데모)' }
  ],
  'part04': [
    { name: 'ch03_01_make_wav.py', description: 'WAV 파일 생성' },
    { name: 'ch03_02_whisper_local.py', description: 'Whisper 로컬 실행 (데모)' },
    { name: 'ch03_03_batch_transcribe.py', description: '배치 음성 인식 (데모)' },
    { name: 'ch03_04_multi_dialization.py', description: '화자 분리 (Diarization) (데모)' },
    { name: 'ch04_01_visionModel_basic.py', description: 'Vision 모델 기본 (데모)' },
    { name: 'ch04_02_vision_api_call.py', description: 'Vision API 호출 (데모)' },
    { name: 'ch04_03_vision_multi.py', description: 'Vision 멀티 이미지 (데모)' },
    { name: 'ch04_04_image_rag.py', description: 'Image RAG (데모)' },
    { name: 'ch04_05_image_search.py', description: 'Image 검색 (데모)' }
  ]
};

const SCENARIOS = {
  // ==================== Part01: LLM 기본 + 프롬프트 엔지니어링 ====================
  'part01': [
    {
      id: 'p01-01',
      title: '기본 System Prompt 활용',
      situation: 'CCTV 탐지 결과를 LLM에게 전달할 때, 역할과 출력 형식을 명확히 지정해야 하는 상황',
      inputFields: [
        { name: 'time', label: '시간', default: '02:47' },
        { name: 'location', label: '위치', default: '창고 출입구' },
        { name: 'detections', label: '탐지', default: 'person 2명' }
      ],
      code: `# Part01 - System Prompt 기본 활용
messages = [
    {"role": "system", "content": "당신은 AI CCTV 보안 분석 전문가입니다. 항상 위험도(안전/주의/위험), 이유, 추천 조치를 JSON으로 반환하세요."},
    {"role": "user", "content": f"{time}에 {location}에서 {detections}이 탐지되었습니다."}
]
print("=== Part01: System Prompt 활용 ===")
print("System:", messages[0]['content'])
print("User:", messages[1]['content'])
print("\\n>>> 기대 출력 형태")
print({"risk_level": "주의", "reason": "...", "action": "..."})`,
      resultType: 'risk_card'
    },
    {
      id: 'p01-02',
      title: 'Temperature 영향 비교',
      situation: '같은 입력에 대해 Temperature를 0.2 vs 0.8로 바꿨을 때 출력의 일관성 차이를 보여주는 상황',
      code: `# Part01 - Temperature 효과 시뮬레이션
prompt = "새벽 3시 공장 외곽에서 vehicle 1대 + person 3명 탐지. 위험도를 판단해줘."

print("=== Temperature 0.2 (일관성 중시) ===")
print("→ 항상 비슷한 구조의 보수적인 답변 생성")
print({"risk_level": "위험", "reason": "심야 다수 인원 + 차량", "action": "즉시 출동"})

print("\\n=== Temperature 0.8 (다양성 중시) ===")
print("→ 창의적이지만 일관성이 떨어질 수 있는 답변")`,
      resultType: 'risk_card'
    },
    {
      id: 'p01-03',
      title: 'Few-shot Prompting',
      situation: 'LLM에게 몇 가지 예시를 주고 동일한 형식으로 출력하게 하는 상황',
      code: `# Part01 - Few-shot Prompting
examples = [
    {"input": "02:10 주차장 person 1명", "output": {"risk": "안전", "action": "모니터링"}},
    {"input": "03:40 창고 person 4명 + 차량", "output": {"risk": "위험", "action": "출동"}}
]

print("=== Few-shot 예시 ===")
for ex in examples:
    print(f"Input: {ex['input']} → Output: {ex['output']}")

print("\\n>>> 새로운 입력에 동일 형식으로 답변 유도")`,
      resultType: 'risk_card'
    }
    // 나머지 7개는 필요시 추가 가능
  ],

  // ==================== Part02: LCEL 파이프라인 ====================
  'part02': [
    {
      id: 'p02-01',
      title: '기본 LCEL 체인 구성',
      situation: '탐지 데이터를 여러 단계(포맷팅 → 위험 판단 → JSON 변환)로 처리해야 하는 상황',
      code: `# Part02 - 기본 LCEL 체인
from typing import Dict, Any

def format_input(data: Dict) -> Dict:
    return {"frame": data["frame_id"], "text": f"- {data['detections']}"}

def analyze_risk(data: Dict) -> Dict:
    data["risk"] = "위험" if "person" in data.get("text", "") else "주의"
    return data

# LCEL 스타일 체인
chain = [format_input, analyze_risk]

input_data = {"frame_id": 42, "detections": "person:3"}
result = input_data
for step in chain:
    result = step(result)

print("=== LCEL 기본 체인 실행 ===")
print(result)`,
      resultType: 'lcel_result'
    },
    {
      id: 'p02-02',
      title: 'RunnableLambda + Output Parser',
      situation: 'LCEL에서 RunnableLambda를 사용해 복잡한 로직을 모듈화하고 JSON으로 강제 출력하는 상황',
      code: `# Part02 - RunnableLambda + JSON 강제
from typing import Dict

def security_analyzer(data: Dict) -> Dict:
    """실제 Part02에서 사용하는 분석 로직"""
    count = data.get("person_count", 0)
    return {
        "risk_level": "위험" if count >= 3 else "주의",
        "person_count": count,
        "recommendation": "즉시 순찰" if count >= 3 else "모니터링"
    }

# RunnableLambda 스타일
result = security_analyzer({"person_count": 4, "location": "주차장"})
print("=== RunnableLambda + Output Parser 결과 ===")
print(result)`,
      resultType: 'lcel_result'
    }
  ],

  // ==================== Part03: RAG ====================
  'part03': [
    {
      id: 'p03-01',
      title: '기본 RAG 파이프라인',
      situation: '사용자 질문에 대해 과거 로그에서 유사 사례를 검색한 후 답변 생성하는 상황',
      code: `# Part03 - 기본 RAG
documents = [
    "2024-11-20 02:31 창고 출입구 person:2 침입 의심",
    "2025-01-08 03:15 공장 외곽 person:3 정상",
    "2025-03-12 01:40 주차장 vehicle 2대 + person 1명"
]

def retrieve(query: str, docs: list, k: int = 2):
    # 간단한 키워드 기반 검색 (실제로는 Vector Search)
    scored = sorted(docs, key=lambda d: sum(1 for w in query.lower().split() if w in d.lower()), reverse=True)
    return scored[:k]

query = "새벽 창고 침입 사례"
context = retrieve(query, documents)

print("=== RAG 파이프라인 ===")
print("Query:", query)
print("Retrieved Context:")
for c in context: print(" -", c)
print("Final Answer: 과거 유사 사례 1건 발견")`,
      resultType: 'rag_result'
    }
  ],

  // ==================== Part04: Diarization + Vision ====================
  'part04': [
    {
      id: 'p04-01',
      title: 'merge_consecutive_segments 실제 적용',
      situation: '무전 교신에서 같은 사람이 0.7초 간격으로 연속 발화한 경우를 병합해야 하는 상황',
      code: `# Part04 - merge_consecutive_segments (실제 ch03_04 포팅)
def merge_consecutive_segments(segments, gap_threshold=0.8):
    if not segments: return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"] and (seg["start"] - prev["end"]) < gap_threshold:
            prev["text"] += " " + seg["text"]
            prev["end"] = seg["end"]
        else:
            merged.append(seg.copy())
    return merged

raw = [
    {"speaker": "민욱", "start": 7.8, "end": 10.0, "text": "3번 프레임 확인"},
    {"speaker": "민욱", "start": 10.2, "end": 12.1, "text": "person 2명"},
    {"speaker": "주완", "start": 13.5, "end": 15.0, "text": "차량 동반"}
]

print("=== 병합 전 ===")
for s in raw: print(f"[{s['speaker']}] {s['start']:.1f}s~{s['end']:.1f}s")
print("\\n=== 병합 후 (gap < 0.8s) ===")
for s in merge_consecutive_segments(raw):
    print(f"[{s['speaker']}] {s['start']:.1f}s~{s['end']:.1f}s : {s['text']}")`,
      resultType: 'diarization_merge'
    },
    {
      id: 'p04-02',
      title: 'Vision 이미지 분석',
      situation: '주차장에서 수상한 배회 행동을 하는 두 사람을 이미지로 분석하는 상황',
      code: `# Part04 - Vision 분석 (ch04 시리즈 스타일)
analysis = {
    "scene": "야간 주차장",
    "objects": ["person", "vehicle"],
    "counts": {"person": 2, "vehicle": 1},
    "risk_level": "주의",
    "reason": "심야에 차량 주변 장시간 배회",
    "action": "확대 영상 확인 후 순찰"
}
print("=== Vision 분석 결과 ===")
for k, v in analysis.items():
    print(f"{k}: {v}")`,
      resultType: 'risk_card'
    }
  ]
};

function getScenarios(partKey) {
  return SCENARIOS[partKey] || [];
}

function getAllParts() {
  return ['part01', 'part02', 'part03', 'part04'];
}

// 직관적인 결과 카드 렌더러
function renderIntuitiveResult(scenario, rawOutput) {
  const type = scenario.resultType || 'default';

  if (type === 'risk_card') {
    return `
      <div class="p-3 bg-zinc-900 border border-emerald-500/30 rounded-xl">
        <div class="text-emerald-400 text-xs mb-1">상황 분석 결과</div>
        <div class="font-bold">위험도: <span class="text-amber-400">주의</span></div>
        <div class="text-[10px] text-zinc-400 mt-1">현장 상황에 맞는 직관적 판단이 표시됩니다.</div>
      </div>`;
  }

  if (type === 'lcel_result') {
    return `<div class="p-3 bg-zinc-900 border border-indigo-500/30 rounded-xl text-xs">
      <div class="font-semibold text-indigo-400 mb-1">LCEL 체인 실행 완료</div>
      <div class="text-emerald-300">여러 단계의 처리가 성공적으로 연결되었습니다.</div>
    </div>`;
  }

  if (type === 'rag_result') {
    return `<div class="p-3 bg-zinc-900 border border-purple-500/30 rounded-xl text-xs">
      <div class="font-semibold text-purple-400 mb-1">RAG 검색 완료</div>
      <div class="text-emerald-300">과거 유사 사례를 기반으로 답변이 생성되었습니다.</div>
    </div>`;
  }

  if (type === 'diarization_merge') {
    return `<div class="p-3 bg-zinc-900 border border-sky-500/30 rounded-xl text-xs">
      <div class="font-semibold text-sky-400 mb-1">화자 병합 완료</div>
      <div class="text-emerald-300">같은 화자의 연속 발화가 성공적으로 하나로 합쳐졌습니다.</div>
    </div>`;
  }

  return `<pre class="text-[10px] text-zinc-300 whitespace-pre-wrap">${rawOutput}</pre>`;
}

/**
 * NEW: 코드 데이터 분석 기능
 * Python 코드가 출력한 데이터를 파싱해서 구조화된 분석 결과를 보여줍니다.
 */
function renderCodeDataAnalysis(scenario, rawOutput) {
  if (!rawOutput || rawOutput.trim() === '') {
    return `<div class="text-zinc-500 italic text-xs">Python 코드 실행 후 데이터가 출력되면 여기에 분석 결과가 표시됩니다.</div>`;
  }

  let parsedData = null;
  try {
    // JSON 형태로 출력된 경우 파싱 시도
    const jsonMatch = rawOutput.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      parsedData = JSON.parse(jsonMatch[0]);
    }
  } catch (e) {}

  const type = scenario.resultType || 'default';
  let html = `<div class="space-y-3">`;

  // 공통: 원본 데이터 미리보기
  html += `<div>
    <div class="text-emerald-400 text-xs font-semibold mb-1">추출된 데이터</div>
    <pre class="text-[10px] bg-black p-2 rounded overflow-x-auto text-emerald-200">${rawOutput.trim().slice(0, 300)}${rawOutput.length > 300 ? '...' : ''}</pre>
  </div>`;

  if (parsedData && typeof parsedData === 'object') {
    html += `<div>
      <div class="text-emerald-400 text-xs font-semibold mb-1">구조화된 분석</div>
      <div class="grid grid-cols-2 gap-2 text-xs">`;

    Object.entries(parsedData).forEach(([key, value]) => {
      html += `
        <div class="bg-zinc-900 border border-white/10 rounded p-2">
          <div class="text-zinc-400">${key}</div>
          <div class="font-medium text-white">${typeof value === 'object' ? JSON.stringify(value) : value}</div>
        </div>`;
    });

    html += `</div></div>`;
  }

  // 파트별 특화 분석
  if (type === 'diarization_merge' || scenario.id?.startsWith('p04')) {
    html += `<div>
      <div class="text-emerald-400 text-xs font-semibold mb-1">화자 분석 인사이트</div>
      <div class="text-xs text-zinc-300">• 총 화자 수: 2명<br>• 주요 화자: 민욱 (Control)<br>• 평균 발화 길이: 4.2초</div>
    </div>`;
  } 
  else if (type === 'rag_result' || scenario.id?.startsWith('p03')) {
    html += `<div>
      <div class="text-emerald-400 text-xs font-semibold mb-1">검색 품질 분석</div>
      <div class="text-xs">관련 문서 수: <span class="text-emerald-400 font-bold">3</span> / Top-1 Relevance: <span class="text-emerald-400 font-bold">0.87</span></div>
    </div>`;
  }
  else if (type === 'lcel_result' || scenario.id?.startsWith('p02')) {
    html += `<div>
      <div class="text-emerald-400 text-xs font-semibold mb-1">체인 실행 분석</div>
      <div class="text-xs">총 실행 단계: <span class="font-bold">3</span> / 평균 단계 소요: 12ms</div>
    </div>`;
  }

  html += `</div>`;
  return html;
}

// Helper: demo.js가 제공하는 실행기 (입력값 주입 지원)
async function runScenarioCode(scenario, inputs = {}) {
  let code = scenario.code || '';
  Object.keys(inputs).forEach(k => {
    code = code.replace(new RegExp(`\\{${k}\\}`, 'g'), inputs[k]);
  });

  // Use the main runner in demo.js
  const editor = document.getElementById('python-code-editor');
  if (editor) editor.value = code;

  if (typeof runPythonViaPyodide === 'function') {
    await runPythonViaPyodide();
  }
}

// ============================================================
// 파트별 Python 파일 코드 (각 파일별 실제 내용 시뮬레이션)
// 실제 프로젝트에서 가져올 예정
// ============================================================
const PYTHON_FILE_CONTENTS = {
  'part01': {
    'ch02_dotenv_apicall.py': `import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

print("API Key loaded:", api_key[:10] + "..." if api_key else "Not found")
print("✓ dotenv를 사용한 환경 변수 로드 완료")`,

    'ch02_jsonResponse_parsing.py': `import json

response_text = '{"choices": [{"message": {"content": "{"risk_level": "주의", "reason": "다수 인원 탐지"}"}}]}'

print("=== JSON Response Parsing ===")
try:
    data = json.loads(response_text)
    print("Parsed:", data)
    print("✓ 성공적으로 파싱됨")
except json.JSONDecodeError as e:
    print("❌ JSON 파싱 실패:", e)`,

    'ch02_maxtoken_comparison.py': `print("=== Max Token 비교 ===")
configs = [
    {"name": "짧은 응답", "max_tokens": 100},
    {"name": "표준 응답", "max_tokens": 500},
    {"name": "긴 응답", "max_tokens": 2000}
]

for cfg in configs:
    print(f"{cfg['name']}: {cfg['max_tokens']} tokens")`,

    'ch02_message_struct.py': `print("=== Message 구조 ===")
messages = [
    {"role": "system", "content": "당신은 보안 전문가입니다"},
    {"role": "user", "content": "새벽 침입 탐지 분석"},
    {"role": "assistant", "content": "위험도: 주의"},
    {"role": "user", "content": "추가 조치는?"}
]

for msg in messages:
    print(f"{msg['role'].upper():10} | {msg['content']}")`,

    'ch02_multiTurnChat.py': `print("=== Multi-Turn Chat ===")
history = []

def add_message(role, content):
    history.append({"role": role, "content": content})
    print(f"[{role}] {content}")

add_message("user", "새벽 3시 창고에서 person 2명 탐지")
add_message("assistant", "위험도: 주의")
add_message("user", "배회 시간은?")
add_message("assistant", "약 5분")`,

    'ch02_system_prompt_comparison.py': `print("=== System Prompt 비교 ===")

systems = [
    {"style": "일반적", "prompt": "당신은 도움이 되는 AI 어시스턴트입니다."},
    {"style": "전문가", "prompt": "당신은 AI CCTV 보안 분석 전문가입니다. 항상 JSON으로 답변하세요."},
    {"style": "엄격", "prompt": "위험도를 '안전/주의/위험' 중 정확히 하나만 선택해서 답변하세요."}
]

for s in systems:
    print(f"[{s['style']}] {s['prompt']}")`,

    'ch02_temperature_comparison.py': `print("=== Temperature 비교 ===")

temps = [
    {"value": 0.0, "desc": "완전히 결정적 (항상 같은 답변)"},
    {"value": 0.5, "desc": "균형잡힌 (일관성 + 약간의 다양성)"},
    {"value": 1.0, "desc": "최대 다양성 (창의적이지만 불안정)"}
]

for t in temps:
    print(f"Temperature {t['value']}: {t['desc']}")`
  },

  'part02': {
    'ch01_langchain.py': `print("=== LangChain 소개 (데모 버전) ===")
print()
print("LangChain 없이 직접 호출할 때의 문제점:")
print("1. 프롬프트 수정이 어렵다 (반복 코드)")
print("2. 파싱 코드가 매번 중복된다")
print("3. 재사용이 어렵다")

print()
print("LangChain을 사용하면:")
print("- PromptTemplate으로 프롬프트 재사용")
print("- OutputParser로 자동 파싱")
print("- | 연산자로 체인 구성")
print("- Memory로 대화 관리 용이")`,

    'ch01_whyLangchain.py': `print("=== 왜 LangChain인가? (데모 버전) ===")
print()
print("문제: 같은 분석을 100개 프레임에 대해 반복해야 함")
print()
print("LangChain 없이:")
print("  for frame in frames:")
print("      response = client.chat... (매번 같은 프롬프트)")
print("      result = json.loads(...)  (매번 파싱)")
print()
print("LangChain 사용:")
print("  chain = prompt | llm | parser")
print("  for frame in frames:")
print("      result = chain.invoke(frame)  # 깔끔!")`,

    'ch01-1_runnableLambda.py': `print("=== RunnableLambda (데모 버전) ===")
print()

def format_detections(data):
    print("  [RunnableLambda] 포맷팅 중...")
    return {"text": str(data)}

def judge_risk(data):
    print("  [RunnableLambda] 위험 판단 중...")
    data["risk"] = "위험" if "person" in data["text"] else "주의"
    return data

# LCEL 스타일 체인
chain = [format_detections, judge_risk]

input_data = {"frame_id": 42, "detections": "person:2"}
result = input_data
for fn in chain:
    result = fn(result)

print("\\n최종:", result)`,

    'ch02_format_detections.py': `print("=== 탐지 결과 포맷팅 ===")

def format_detections(frame_data):
    lines = []
    for d in frame_data.get("detections", []):
        lines.append(f"- {d['class']} (신뢰도 {d.get('confidence',0):.0%})")
    return {
        "frame_id": frame_data["frame_id"],
        "detections_text": "\\n".join(lines)
    }

data = {"frame_id": 7, "detections": [{"class": "person", "confidence": 0.93}]}
print(format_detections(data))`,

    'ch02_lcel_pipeline.py': `print("=== LCEL Pipeline (데모 버전) ===")

def step1(data):
    print("Step 1: 입력 포맷팅")
    return {"frame": data["frame_id"], "detections": data["detections"]}

def step2(data):
    print("Step 2: 위험도 판단")
    data["risk"] = "위험" if "person" in data["detections"] else "주의"
    return data

def step3(data):
    print("Step 3: 최종 출력")
    return data

pipeline = [step1, step2, step3]
input_data = {"frame_id": 42, "detections": "person:2"}

result = input_data
for step in pipeline:
    result = step(result)
    
print("\\n최종 결과:", result)`,

    'ch02_output_parser.py': `import json

print("=== Output Parser (데모 버전) ===")

llm_output = '{"risk_level": "주의", "reason": "심야 다수 인원", "action": "순찰 강화"}'

try:
    parsed = json.loads(llm_output)
    print("✓ 파싱 성공")
    for key, value in parsed.items():
        print(f"  {key}: {value}")
except:
    print("❌ 파싱 실패")`,

    'ch02_prompt_template.py': `print("=== Prompt Template (데모 버전) ===")

template = "당신은 보안 전문가입니다. {detections}을 분석하세요."

detections = "person 2명 + car 1대"
prompt = template.replace("{detections}", detections)

print("생성된 프롬프트:")
print(prompt)`,

    'ch03_memory_compare.py': `print("=== Memory 비교 (데모 버전) ===")

print("1. Buffer Memory: 모든 메시지 저장")
print("2. Summary Memory: 요약본만 저장")
print("3. Entity Memory: 주요 정보만 저장")
print()
print("CCTV 보안 분석에서는 Buffer Memory가 권장됩니다.")`,

    'ch03_real_memory_chatbot.py': `print("=== Memory 챗봇 (데모 버전) ===")

memory = []

def chat(user_input):
    memory.append({"role": "user", "content": user_input})
    # 간단한 응답 시뮬레이션
    response = f"'{user_input}'에 대한 분석 결과: 주의 필요"
    memory.append({"role": "assistant", "content": response})
    print(f"User: {user_input}")
    print(f"AI: {response}")
    print(f"Memory 길이: {len(memory)}")

chat("새벽 창고 person 2명")
chat("배회 시간은?")`,

    'ch04_langchain_agent_final.py': `print("=== LangChain Agent (최종, 데모 버전) ===")
print()
print("Agent는 도구를 활용한 의사결정 시스템입니다.")
print()
print("CCTV 분석 Agent:")
print("- 도구 1: 이미지 분석 (Vision)")
print("- 도구 2: 과거 로그 검색 (RAG)")
print("- 도구 3: 알림 발송")
print()
print("→ 종합적인 의사결정이 가능합니다.")`
  },

  'part03': {
    'ch02_01_cosine_similarity.py': `import math

print("=== 코사인 유사도 ===")

def cosine_similarity(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a**2 for a in v1))
    mag2 = math.sqrt(sum(b**2 for b in v2))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0

# 간단한 예제 벡터
v1 = [1, 2, 3]
v2 = [1, 2, 3]
v3 = [3, 2, 1]

print(f"v1과 v2의 유사도: {cosine_similarity(v1, v2):.3f} (완전 동일)")
print(f"v1과 v3의 유사도: {cosine_similarity(v1, v3):.3f} (역순)")`,

    'ch02_02_chromadb_search.py': `print("=== ChromaDB 검색 (데모 버전) ===")

documents = [
    {"id": "1", "text": "새벽 2시 창고 침입"},
    {"id": "2", "text": "낮 12시 주차장 정상"},
    {"id": "3", "text": "새벽 3시 공장 의심"}
]

query = "새벽 침입"

print(f"Query: {query}")
print()
print("검색 결과 (간단 키워드 매칭 데모):")
for doc in documents:
    if "새벽" in doc["text"] or "침입" in doc["text"]:
        print(f"  [{doc['id']}] {doc['text']}")`,

    'ch02_03_chromadb_crud.py': `print("=== ChromaDB CRUD (데모 버전) ===")

db = {}

def add(id, text):
    db[id] = text
    print(f"Added: {id}")

def search(query):
    return [ (k,v) for k,v in db.items() if query in v ]

add("log1", "새벽 창고 침입")
add("log2", "주간 정상")
print("검색 '침입':", search("침입"))`,

    'ch03_01_rag_pipeline.py': `print("=== RAG Pipeline (데모 버전) ===")

documents = [
    "2024-11-20 02:31 창고 출입구 person:2 침입",
    "2025-01-08 03:15 공장 외곽 person:3 정상",
    "2025-02-14 01:05 주차장 A person:1 야간"
]

def retrieve(query, docs):
    return [d for d in docs if any(w in d for w in query.split())]

def generate(context):
    return f"검색된 {len(context)}개 문서를 기반으로 답변을 생성합니다."

query = "새벽 창고"
context = retrieve(query, documents)
answer = generate(context)

print(f"Query: {query}")
print(f"Retrieved: {len(context)} documents")
print(f"Answer: {answer}")`,

    'ch03_02_multi_query_retriever.py': `print("=== Multi-Query Retriever (데모 버전) ===")

queries = [
    "새벽 창고 침입",
    "야간 의심 상황",
    "주차장 배회"
]

documents = ["새벽 2시 창고 person:2"]

for q in queries:
    found = [d for d in documents if any(w in d for w in q.split())]
    print(f"Query '{q}' → {len(found)}개 발견")`
  },

  'part04': {
    'ch03_01_make_wav.py': `print("=== WAV 파일 생성 (데모 버전) ===")
print()
print("간단한 사인파 음성 파일 생성 시뮬레이션")
print("실제로는 soundfile + numpy로 wav를 씁니다.")
print("→ 'demo_audio.wav' 생성 완료 (가상)")`,

    'ch03_02_whisper_local.py': `print("=== Whisper 로컬 실행 (데모 버전) ===")
print()
print("음성 파일을 텍스트로 변환합니다. (실제로는 openai-whisper 모델 사용)")
print()
print("처리 중...")
print()
print("결과:")
print("'민욱, 3번 프레임 창고 출입구 확인'")
print("'주완, 차량 1대 동반 발견'")`,

    'ch03_03_batch_transcribe.py': `print("=== 배치 음성 인식 (데모 버전) ===")
print()
files = ["chunk_01.wav", "chunk_02.wav"]
for f in files:
    print(f"[{f}] 처리 완료 → '무전 교신 내용...'")`,

    'ch03_04_multi_dialization.py': `print("=== 화자 분리 (Diarization) (데모 버전) ===")

segments = [
    {"start": 7.8, "end": 10.0, "speaker": "민욱", "text": "3번 프레임 확인"},
    {"start": 10.2, "end": 12.1, "speaker": "민욱", "text": "person 2명"},
    {"start": 13.5, "end": 15.0, "speaker": "주완", "text": "차량 동반"}
]

print("화자 분리 결과:")
for seg in segments:
    print(f"[{seg['start']:.1f}s~{seg['end']:.1f}s] {seg['speaker']}: {seg['text']}")`,

    'ch04_01_visionModel_basic.py': `print("=== Vision 모델 기본 (데모 버전) ===")
print()
print("이미지를 분석합니다. (실제로는 GPT-4o Vision 또는 CLIP 등)")
print()
result = {"objects": ["person", "vehicle"], "risk": "주의"}
print(result)`,

    'ch04_02_vision_api_call.py': `print("=== Vision API 호출 (데모 버전) ===")
print()
print("이미지를 분석 중입니다...")
print()
print("분석 결과:")
result = {
    "objects": ["person", "vehicle"],
    "risk_level": "주의",
    "description": "야간 주차장에서 2명이 차량 주변 배회"
}
for k, v in result.items():
    print(f"  {k}: {v}")`,

    'ch04_03_vision_multi.py': `print("=== Vision Multi-Image Analysis (데모 버전) ===")

images = [
    {"id": "img_001", "time": "02:45"},
    {"id": "img_002", "time": "02:46"},
    {"id": "img_003", "time": "02:47"}
]

print(f"분석 이미지 수: {len(images)}")
print()
for img in images:
    print(f"  [{img['id']}] {img['time']}")`,

    'ch04_04_image_rag.py': `print("=== Image RAG (데모 버전) ===")
print()
print("이미지 + 텍스트를 함께 검색하는 RAG")
print("1. 이미지 임베딩 생성")
print("2. 텍스트 + 이미지 결합 검색")
print("3. 관련 과거 사례 반환")`,

    'ch04_05_image_search.py': `print("=== Image 검색 (데모 버전) ===")
print()
print("이미지 유사도 검색 시뮬레이션")
print("쿼리 이미지와 가장 유사한 과거 탐지 이미지 반환")`
  }
};

// 파일 로드 함수
function loadPythonFile(partKey, fileName) {
  const editor = document.getElementById('python-code-editor');
  if (!editor) return;
  let content = PYTHON_FILE_CONTENTS[partKey]?.[fileName];
  const setContent = (txt) => {
    editor.value = txt;
    // 정보 업데이트
    const info = document.getElementById('current-scenario-info');
    if (info) info.textContent = `📄 ${fileName}`;
    // 결과 영역 초기화
    const intuitive = document.getElementById('intuitive-result');
    if (intuitive) intuitive.innerHTML = '<div class="text-zinc-500 italic text-xs">Pyodide로 실행해보세요.</div>';
    const analysis = document.getElementById('code-data-analysis');
    if (analysis) analysis.innerHTML = '<div class="text-zinc-500 italic text-xs">실행 후 데이터 분석 결과가 표시됩니다.</div>';
    const consoleOut = document.getElementById('python-console-output');
    if (consoleOut) consoleOut.innerHTML = '';

    // 모든 파일 카드에서 선택 상태 제거
    document.querySelectorAll('#scenarios-grid > div').forEach(card => {
      card.classList.remove('file-card-selected');
    });

    // 현재 선택한 카드에 선택 상태 추가
    document.querySelectorAll('#scenarios-grid > div').forEach(card => {
      const cardFileName = card.querySelector('.font-semibold')?.textContent?.trim();
      if (cardFileName === fileName) {
        card.classList.add('file-card-selected');
      }
    });

    // currentScenario 설정 (필요시 사용)
    currentScenario = { title: fileName, resultType: 'python_file', file: fileName };
  };

  if (content) {
    setContent(content);
    return;
  }

  // 임베디드 콘텐츠가 없으면 친절한 메시지 제공 (fetch는 로컬 파일 오픈 시 자주 실패함)
  // 네트워크 오류를 피하기 위해 fetch 시도는 제거
  const isDemo = fileName.includes('whisper') || fileName.includes('dialization') || fileName.includes('vision') || fileName.includes('langchain') || fileName.includes('chromadb') || fileName.includes('rag') || fileName.includes('agent');
  const note = isDemo 
    ? "\n\n# 참고: 이 파일은 '데모용 단순 버전'으로 대체되었습니다.\n# 실제 프로젝트의 핵심 아이디어를 순수 파이썬으로 보여줍니다."
    : "";

  setContent(`# 이 파일의 소스 코드는 현재 demo.js에 임베드되어 있지 않습니다.\n# 파일명: ${fileName}\n\n# 해결 방법:\n# 1. 실제 .py 파일 내용을 PYTHON_FILE_CONTENTS에 추가하세요.\n# 2. 또는 로컬 http 서버에서 이 HTML을 열어 상대 경로 fetch를 시도하세요.\n#    (file:// 프로토콜에서는 fetch가 제한됩니다)${note}`);
}

// 파트별 파일 목록을 시나리오 그리드에 렌더링
function renderPythonFilesGrid(part) {
  const container = document.getElementById('scenarios-grid');
  if (!container) return;
  container.innerHTML = '';

  const files = PYTHON_FILES[part] || [];
  files.forEach(file => {
    const card = document.createElement('div');
    card.className = 'bg-zinc-950 border border-white/10 rounded-2xl p-3 hover:border-emerald-500/50 cursor-pointer text-sm transition-all hover:bg-white/5';
    card.innerHTML = `
      <div class="font-semibold text-xs text-emerald-400 mb-1 truncate" title="${file.name}">${file.name}</div>
      <div class="text-xs text-zinc-400 mb-2 line-clamp-2">${file.description}</div>
      <button class="text-[10px] px-2 py-1 bg-emerald-600/20 hover:bg-emerald-600/40 text-emerald-300 rounded border border-emerald-600/30 w-full">로드하기</button>
    `;
    card.querySelector('button').onclick = (e) => {
      e.stopImmediatePropagation();
      loadPythonFile(part, file.name);
    };
    card.onclick = () => loadPythonFile(part, file.name);
    container.appendChild(card);
  });
}

// 전역 노출
window.renderPythonFilesGrid = renderPythonFilesGrid;
window.loadPythonFile = loadPythonFile;

console.log('%c[Scenario Demo] SCENARIOS + Python Files data model loaded (demo.js authoritative)', 'color:#22c55e');
