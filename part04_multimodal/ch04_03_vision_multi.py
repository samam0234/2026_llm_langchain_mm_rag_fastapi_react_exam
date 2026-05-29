import json, os, re, base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 한번의 api호출로 이미지 여러장을 분석 시키기
from ch04_01_visionModel_basic import image_to_base64
from ch04_02_vision_api_call import json_parse

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# ────────────────── analyze_multiple_images 변수 만들어 analyze 호출시키기 ────────────────────────────

def analyze_multiple_images(image_path:list[str] , prompt: str) -> dict : 
    """
    이미지 여러 장을 한 번의 API 호출로 비교 분석한다.

    핵심 아이디어:
        content 리스트에 이미지 블록을 여러 개 넣으면
        LLM이 모든 이미지를 동시에 보면서 비교·분석한다.

        [이미지1 블록, 이미지2 블록, 이미지3 블록, 텍스트 블록]

    Args:
        image_paths : 분석할 이미지 파일 경로 리스트
        prompt      : 분석 지시
    """
    
    # 이미지 블럭들을 리스트로 조합
    content_blocks = []
    for i, path in enumerate(image_paths, start=1) :
       image_b64, media_type = image_to_base64(path)
       b64, mt = image_to_base64(path)
       content_blocks.append(
           {"type":  "image_url", "image_url": {
                    "url": f"data:{mt};base64,{b64}",
                    # detail 옵션:
                    #   "low"  → 빠름, 저렴, 간단한 분류에 적합
                    #   "high" → 느림, 비쌈, 세부 내용 분석에 적합
                    "detail" : "low"
            }},  # 이미지 블록
           
       )
    content_blocks.append({"type": "text", "text": prompt}) # 텍스트 블록
       
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": content_blocks}
        ],
        max_tokens=500
    )
    
    return json_parse(response.choices[0].message.content)


# ── 실행 ──────────────────────────────────────────────────────
image_paths = ["./vision_sample/cat.jpeg", "./vision_sample/Golden_retriever.jpeg", "./vision_sample/Blue_parrot.jpeg"]

prompt = """위 이미지들을 순서대로 분석해서 아래 JSON 형식으로만 응답하세요.
{
  "images": [
    {"index": 1, "subject": "피사체", "description": "설명"},
    {"index": 2, "subject": "피사체", "description": "설명"},
    {"index": 3, "subject": "피사체", "description": "설명"}
  ],
  "common_theme": "세 이미지의 공통 주제"
}"""

print("이미지 로딩 중...")
result = analyze_multiple_images(image_paths, prompt)
print("\n=== 분석 결과 ===")
print(json.dumps(result, ensure_ascii=False, indent=2))