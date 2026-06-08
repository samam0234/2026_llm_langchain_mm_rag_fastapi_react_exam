# ch03_04_multi_dialization.py 에서 배운 내용을 기반으로 연습을 할 것
# 다만 여기서 추가로 tts도 같이 해볼 것임
# 여기서 사용할 tts 모델은 coqui-ai의 TTS 모델로, Hugging Face에서 "coqui-ai/tts_models"로 검색하면 다양한 모델이 나옴
# 그 중에서 "coqui-ai/tts_models/en/ljspeech/tacotron2-DDC" 모델을 사용할 예정임
# 이 모델은 영어 TTS 모델이지만, 한국어 발음도 어느 정도 지원하므로 연습용으로 사용하기에 적합함
# TTS 모델을 사용하기 위해서는 먼저 Hugging Face에서 모델을 다운로드 받아야 함
# Hugging Face에서 모델을 다운로드 받으려면 Hugging Face 계정이 필요하며, 계정을 만든 후에는 "Settings" → "Access Tokens"에서 "New token"을 생성하여 토큰을 발급받아야 함 (이미 발급받았다면 그 토큰을 사용하면 됨)
# 발급받은 토큰은 .env 파일에 HF_TOKEN=발급받은_토큰 형식으로 저장해야 함 <- 완료된 .env 파일 예시는 위에 있음
# TTS 모델을 사용하기 위해서는 transformers 라이브러리를 설치해야 함 (설치 명령어: pip install transformers)
# transformers 라이브러리를 설치한 후에는 아래와 같이 TTS 모델을 로드할 수 있음

# ============================================================
# [중요] STEP 5~6 (LLM 요약 + TTS)을 실행하려면 아래 패키지도 필요합니다.
# pip install gtts
#
# 전체 프로젝트 의존성은 requirments.txt 를 참고하세요.
# ============================================================

import time

import os
import json
import re
import soundfile as sf
import torch
from dotenv import load_dotenv

import whisper
from pyannote.audio import Pipeline

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─── 중요: import 전에 load_dotenv() 먼저 실행 ─────────────────
load_dotenv()

# ─── 로컬 설정 (ch03_04_multi_dialization.py 없이 독립적으로 실행) ─────────────────
# ch03_04_multi_dialization.py 파일에 문제가 있으므로, 여기서 필요한 설정을 직접 정의합니다.

MODEL_NAME = "base"               # CPU: "base" / GPU: "turbo"
AUDIO_PATH = "./20260526_All_units.wav"
HF_TOKEN = os.getenv("HF_TOKEN")

# 화자 이름 매핑
SPEAKER_NAMES = {
    "SPEAKER_00": "민욱 (Control)",
    "SPEAKER_01": "주완 (Unit 3)",
    "SPEAKER_02": "대진 (Unit 2)",
    "SPEAKER_03": "다은 (Unit 5)",
}

def correct_timestanps(segments: list, actual_duration: float) -> tuple:
    """
    Whisper 타임스탬프 보정 함수.
    """
    if not segments:
        return segments, 1.0
    
    whisper_max = segments[-1]["end"]
    
    if abs(whisper_max - actual_duration) < 1.0:
        return segments, 1.0
    
    ratio = actual_duration / whisper_max
    print(f"  ⚠️ 타임스탬프 보정: {whisper_max:.1f}s → {actual_duration:.1f}s (비율: {ratio:.4f})")
    
    corrected = []
    for seg in segments:
        s = seg.copy()
        s["start"] = round(seg["start"] * ratio, 2)
        s["end"] = round(seg["end"] * ratio, 2)
        corrected.append(s)
    
    return corrected, ratio

# ── 새로 추가되는 TTS & LangChain 설정 ───────────────────────
TTS_ENGINE = "gTTS"                    # "gTTS" or "coqui" or "pyttsx3" (추후 변경 용이)
TTS_LANGUAGE = "en"                    # "en" (영어 무전) / "ko" (한국어 요약)
TTS_OUTPUT_PATH = "./20260526_All_units_tts_response.mp3"

# LangChain 설정
LLM_MODEL_NAME = "gpt-4o-mini"         # 비용/성능 균형 좋은 모델
LLM_TEMPERATURE = 0.3                  # 요약 시 창의성 낮춤 (사실 중심)
MAX_TOKENS = 500

# 출력 파일명 (전사 + TTS)
TRANSCRIPT_PATH = AUDIO_PATH.replace(".wav", "_transcript.txt")
TTS_RESPONSE_PATH = AUDIO_PATH.replace(".wav", "_tts_response.mp3")

# 초기 프롬프트 (LLM이 무전 내용을 자연스럽게 요약하게 함)
INITIAL_SUMMARY_PROMPT = """
당신은 군사 무전 통신을 요약하는 전문가입니다.
다음은 여러 화자가 참여하는 무전 대화입니다.
각 화자의 발언을 구분하여 요약해 주세요.
- 발언은 화자 이름과 함께 구분되어 있습니다.
"""

print("=" * 60)
print("✅ 설정 로드 완료 (STEP 1 테스트 모드)")
print(f"   오디오 파일     : {AUDIO_PATH}")
print(f"   HF_TOKEN        : {'있음' if HF_TOKEN else '없음'}")
print("=" * 60)

# === STEP 1 테스트를 위한 정리된 영역 ===

# ─────────────────────────────────────────────────────────────
# STEP 1: Whisper STT + Pyannote 화자 분리 (확정 구현)
# ─────────────────────────────────────────────────────────────
 
def step1_whisper_and_diarization(
    audio_path: str,
    model_name: str = "base",
    hf_token: str = None,
    language: str = "en",
    min_speakers: int = 2,
    max_speakers: int = 4,
) -> dict:
    """
    Whisper 음성 인식 + Pyannote 화자 분리를 한 번에 수행하는 함수.
 
    반환값 (dict):
        {
            "result": whisper transcribe 결과 (dict),
            "annotation": pyannote Annotation 객체,
            "actual_duration": float (초),
            "raw_segments": list (보정 전/후 segments),
            "audio_path": str
        }
    """
    print("=" * 60)
    print("STEP 1: Whisper STT + Pyannote 화자 분리")
    print("=" * 60)
 
    # ============================================
    # 1-1. Whisper STT
    # ============================================
    print("\n[1/2] Whisper 모델 로드 및 음성 인식 시작...")
    whisper_model = whisper.load_model(model_name)
 
    result = whisper_model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        initial_prompt=(
            "Security radio communication. "
            "Units responding to suspicious activity at parking structure."
        ),
        verbose=False,
        fp16=False,          # CPU 환경에서는 반드시 False
    )
 
    print(f"  ✅ Whisper STT 완료 — 세그먼트 수: {len(result['segments'])}개")
    print(f"     전체 텍스트 미리보기: {result['text'][:80]}...\n")
 
    # 실제 오디오 길이 가져오기 (타임스탬프 보정용)
    actual_duration = sf.info(audio_path).duration
    print(f"  실제 오디오 길이: {actual_duration:.2f}초")
 
    # Whisper 타임스탬프 보정 (Windows ffmpeg 문제 대응)
    corrected_segments, ratio = correct_timestanps(result["segments"], actual_duration)
    result["segments"] = corrected_segments   # 보정된 segments로 교체
 
    # ============================================
    # 1-2. Pyannote 화자 분리
    # ============================================
    print("\n[2/2] Pyannote 화자 분리 모델 로드 및 분석 시작...")
 
    if not hf_token:
        raise ValueError("HF_TOKEN이 필요합니다. .env 파일에 HF_TOKEN을 설정하세요.")
 
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
 
    # Windows 호환성을 위해 soundfile 사용 (torchaudio 대신)
    data, sample_rate = sf.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(data).unsqueeze(0)   # (time,) → (1, time)
 
    # 중요: 실제 파일의 sample_rate를 사용해야 pyannote가 올바른 타임스탬프를 생성합니다.
    actual_sr = sample_rate   # sf.read로 읽은 실제 샘플레이트 사용

    output = diarization_pipeline(
        {"waveform": waveform, "sample_rate": actual_sr},
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
 
    # pyannote 버전에 따라 반환 형식이 다를 수 있음 → 안전하게 추출
    if hasattr(output, "speaker_diarization") and output.speaker_diarization is not None:
        annotation = output.speaker_diarization
    else:
        annotation = output  # fallback for some versions

    # annotation이 제대로 된 객체인지 최종 확인
    if annotation is None or not hasattr(annotation, "itertracks"):
        raise RuntimeError(
            f"pyannote diarization 결과에서 Annotation을 추출할 수 없습니다.\n"
            f"output 타입: {type(output)}\n"
            f"output 내용 일부: {str(output)[:200]}"
        )
 
    print("  ✅ Pyannote 화자 분리 완료")
    print("     감지된 화자 구간:")
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        print(f"       {speaker}: {turn.start:.1f}s ~ {turn.end:.1f}s")
    print()
 
    return {
        "result": result,
        "annotation": annotation,
        "actual_duration": actual_duration,
        "raw_segments": result["segments"],
        "audio_path": audio_path,
    }
 

# ============================================================
# STEP 2: 화자 매핑 + 연속 발화 병합 (구현 시작)
# ============================================================

# ch03_04_multi_dialization.py 에서 배운 핵심 헬퍼 함수들을 Practice 파일에 직접 작성
# (학습 목적으로 복사/구현하는 것을 권장)

def get_speaker_at(annotation, start: float, end: float) -> str:
    """
    Whisper 세그먼트의 중간 시점(mid)이 속하는 화자를 반환합니다.

    중간 시점을 쓰는 이유:
        Whisper 세그먼트와 pyannote 구간의 경계가 정확히 일치하지 않음.
        중간값을 쓰면 경계 오류를 줄일 수 있음.

    < (미만) 비교를 쓰는 이유:
        경계에 걸쳐 있는 경우 중복 매핑을 방지하기 위함.
    """
    if annotation is None:
        return "UNKNOWN"

    # pyannote Annotation 객체인지 간단히 확인 (디버깅용)
    if not hasattr(annotation, "itertracks"):
        raise TypeError(
            f"annotation이 올바른 pyannote.core.Annotation 객체가 아닙니다. "
            f"현재 타입: {type(annotation)}. "
            "step1에서 annotation = output.speaker_diarization 로 제대로 추출했는지 확인하세요."
        )

    mid = (start + end) / 2
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        if turn.start <= mid < turn.end:
            return speaker
    return "UNKNOWN"


def merge_consecutive_segments(segments: list) -> list:
    """
    연속된 같은 화자의 발화를 하나로 합칩니다.

    예시:
        [민욱] "Subject attempted..."  7.8s ~ 10.0s
        [민욱] "Access denied three times." 10.1s ~ 12.0s
        → [민욱] "Subject attempted... Access denied three times." 7.8s ~ 12.0s

    gap_threshold: 0.8초 이내 같은 화자 → 연속 발화로 판단
    """
    if not segments:
        return []

    gap_threshold = 0.8
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


def step2_3_merge_segments(result: dict, annotation) -> list:
    """
    STEP 2 + STEP 3 통합 함수

    1. Whisper 세그먼트에 화자 매핑 (get_speaker_at + SPEAKER_NAMES)
    2. 연속된 같은 화자 발화 병합 (merge_consecutive_segments)

    반환값:
        merged_segments (list of dict)
        각 항목 예시:
            {
                "speaker": "SPEAKER_00",
                "speaker_name": "민욱 (Control)",
                "start": 7.82,
                "end": 12.05,
                "text": "Subject attempted access. Access denied three times."
            }
    """
    print("=" * 60)
    print("STEP 2 + 3: 화자 매핑 + 연속 발화 병합")
    print("=" * 60)

    if annotation is None:
        raise ValueError("annotation이 None입니다. pyannote 화자 분리가 제대로 수행되지 않았습니다.")

    segments_with_speaker = []

    print(f"  → 총 {len(result.get('segments', []))}개 Whisper 세그먼트에 화자 매핑 시작...")

    for idx, seg in enumerate(result["segments"]):
        try:
            speaker_id = get_speaker_at(annotation, seg["start"], seg["end"])
        except Exception as e:
            print(f"  ⚠️ 세그먼트 {idx} 매핑 중 에러 발생: {e}")
            print(f"     seg = {seg}")
            raise

        speaker_name = SPEAKER_NAMES.get(speaker_id, speaker_id)

        segments_with_speaker.append({
            "speaker": speaker_id,
            "speaker_name": speaker_name,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        })

    merged_segments = merge_consecutive_segments(segments_with_speaker)

    print(f"✅ 병합 완료 — {len(result['segments'])}개 세그먼트 → {len(merged_segments)}개\n")

    # 간단한 미리보기 출력 (학습용)
    print("병합된 세그먼트 미리보기 (최대 5개):")
    for i, seg in enumerate(merged_segments[:5]):
        print(f"  [{i+1}] [{seg['speaker_name']}] {seg['start']:.1f}s ~ {seg['end']:.1f}s")
        print(f"       {seg['text'][:80]}{'...' if len(seg['text']) > 80 else ''}")
    if len(merged_segments) > 5:
        print(f"  ... 외 {len(merged_segments) - 5}개")
    print()

    return merged_segments


# ============================================================
# STEP 4: 화자별 대화 전사 출력 + 파일 저장 (구현)
# ============================================================

def step4_save_transcript(merged_segments: list, output_path: str = None) -> str:
    """
    STEP 4: 화자별 대화 전사 출력 및 텍스트 파일 저장

    - 콘솔에 보기 좋게 출력 (MM:SS.ss 형식 + 화자 이름)
    - 지정된 경로에 .txt 파일로 저장

    반환값: 저장된 전사 파일의 경로 (str)
    """
    if output_path is None:
        output_path = TRANSCRIPT_PATH

    print("=" * 60)
    print("STEP 4: 화자별 대화 전사")
    print("=" * 60)
    print()

    # 1. 콘솔에 예쁘게 출력
    for seg in merged_segments:
        # 초 → 분:초 형식 변환
        start_min = int(seg["start"]) // 60
        start_sec = seg["start"] % 60
        end_min   = int(seg["end"])   // 60
        end_sec   = seg["end"]   % 60

        print(f"┌─ [{seg['speaker_name']}]  "
              f"{start_min:02d}:{start_sec:05.2f} → {end_min:02d}:{end_sec:05.2f}")
        print(f"│  {seg['text']}")
        print("│")

    print("└─ (전사 종료)")
    print()

    # 2. 파일로 저장
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== 화자별 대화 전사 ===\n\n")
        for seg in merged_segments:
            f.write(f"[{seg['speaker_name']}] ({seg['start']:.1f}s ~ {seg['end']:.1f}s)\n")
            f.write(f"{seg['text']}\n\n")

    print(f"📄 전사 파일 저장: {output_path}\n")

    return output_path


# ============================================================
# STEP 5: LLM 요약 (LangChain + ChatOpenAI)
# ============================================================

def step5_summarize_with_llm(merged_segments: list) -> str:
    """
    STEP 5: 병합된 화자 대화를 LLM으로 요약합니다.

    - INITIAL_SUMMARY_PROMPT 사용
    - 비용 절감을 위해 gpt-4o-mini + temperature=0.3 사용
    """
    print("=" * 60)
    print("STEP 5: LLM 요약 생성")
    print("=" * 60)

    # 화자별 대화 내용을 읽기 좋은 문자열로 변환
    transcript_text = ""
    for seg in merged_segments:
        transcript_text += f"[{seg['speaker_name']}] {seg['text']}\n"

    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    summary_prompt = PromptTemplate.from_template(INITIAL_SUMMARY_PROMPT + "\n\n{transcript}")

    chain = summary_prompt | llm | StrOutputParser()

    print("  → LLM에게 요약 요청 중...")
    summary = chain.invoke({"transcript": transcript_text.strip()})

    print("\n📝 LLM 요약 결과:")
    print("-" * 50)
    print(summary)
    print("-" * 50 + "\n")

    return summary


# ============================================================
# STEP 6: TTS (Text-to-Speech)
# ============================================================

def step6_tts(text: str, engine: str = None) -> str:
    """
    STEP 6: 요약문을 음성으로 변환하여 저장합니다.

    지원 엔진:
      - "gTTS" (기본, 가볍고 설치 쉬움)
      - "coqui" (고품질, transformers 필요 - 주석 참고)
    """
    if engine is None:
        engine = TTS_ENGINE

    print("=" * 60)
    print(f"STEP 6: TTS 음성 생성 (엔진: {engine})")
    print("=" * 60)

    output_path = TTS_OUTPUT_PATH

    if engine.lower() == "gtts":
        # gTTS는 선택적 패키지입니다.
        # pip install gtts 로 설치해야 합니다.
        try:
            from gtts import gTTS  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "gTTS 패키지가 설치되어 있지 않습니다.\n"
                "해결 방법: pip install gtts"
            )

        print("  → gTTS로 음성 생성 중...")
        tts = gTTS(text=text, lang=TTS_LANGUAGE)
        tts.save(output_path)
        print(f"🔊 TTS 파일 저장 완료: {output_path}\n")
        return output_path

    elif engine.lower() == "coqui":
        print("  → coqui TTS는 아직 구현되지 않았습니다.")
        print("     Practice 파일 상단 주석을 참고하여 transformers + coqui 모델을 추가하세요.")
        # TODO: transformers pipeline으로 coqui 모델 로드 후 synthesis
        return ""

    else:
        print(f"지원하지 않는 TTS 엔진입니다: {engine}")
        return ""


# ============================================================
# 🧪 실험 실행 블록 (Progressive Test - STEP 1 ~ 6 지원)
# ============================================================

if __name__ == "__main__":
    # === 실험 설정 ===
    # 1 = STEP 1만
    # 2 = STEP 1 + 2/3
    # 3 = STEP 1~4 (전사 저장)
    # 4 = STEP 1~5 (LLM 요약)
    # 5 = STEP 1~6 (TTS 음성까지 전체)
    TEST_UP_TO_STEP = 5

    print("\n" + "=" * 60)
    print("🧪 ch03_04_multi_dialization_Practice.py - Full Pipeline Test")
    print("=" * 60 + "\n")

    outputs = step1_whisper_and_diarization(
        audio_path=AUDIO_PATH,
        model_name=MODEL_NAME,
        hf_token=HF_TOKEN,
        language="en",
        min_speakers=2,
        max_speakers=4,
    )

    merged_segments = None
    summary_text = None

    if TEST_UP_TO_STEP >= 2:
        merged_segments = step2_3_merge_segments(
            result=outputs["result"],
            annotation=outputs["annotation"]
        )

    if TEST_UP_TO_STEP >= 3:
        if merged_segments:
            step4_save_transcript(merged_segments, TRANSCRIPT_PATH)

    if TEST_UP_TO_STEP >= 4:
        if merged_segments:
            summary_text = step5_summarize_with_llm(merged_segments)

    if TEST_UP_TO_STEP >= 5:
        if summary_text:
            tts_path = step6_tts(summary_text, engine=TTS_ENGINE)
            if tts_path:
                print(f"✅ 전체 파이프라인 완료! 최종 음성 파일: {tts_path}")

    # 요약 메시지
    print("\n" + "=" * 60)
    if TEST_UP_TO_STEP == 5:
        print("🎉 [STEP 1 ~ 6 전체 파이프라인 테스트 완료]")
    else:
        print(f"✅ [STEP 1 ~ {TEST_UP_TO_STEP} 테스트 완료]")
        print(f"   TEST_UP_TO_STEP = 5 로 변경하면 TTS까지 전체 실행됩니다.")
