# 영구히 저장한 chroma db운용법

import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from ch02_01_cosine_similarity import get_embedding

DB_PATH = "./chroma_db"     # DB경로
COLLECTION_NAME = "cctv_detection_logs"     # 컬렉션 이름

load_dotenv()

client_oai = OpenAI()

# collection 불러오기
# 1. create_collection("컬렉션 이름") - 없으면 생성. 있으면 에러
# 2. get_collection("컬렉션 이름") - 없으면 에러.
# get_or_create_collention("컬렉션 이름") - 없으면 생성. 있으면 가져옴
chroma_client = chromadb.PersistentClient(path=DB_PATH)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, 
    metadata={
        "hnsw:space" : "cosine"     # 텍스트 임베팅에 가장 적합한 거리 측청방식(방향만 보고 크기 무시)
    }
)

print(f"✅ DB 경로: {os.path.abspath(DB_PATH)}")
print(f"✅ 컬렉션: {COLLECTION_NAME} | 현재 저장된 개수: {collection.count()}\n")

if collection.count() == 0 :
    print("=== STEP 1: 초기 데이터 저장 (최초 실행) ===")

    initial_logs = [
        {
            "id": "log_001",
            "text": "새벽 2시 창고 출입구 침입. person:2 car:1. 위험.",
            "metadata": {"risk_level": "위험", "location": "창고 출입구", "resolved": False},
        },
        {
            "id": "log_002",
            "text": "오후 2시 주차장 A. person:1. 정상 이용객.",
            "metadata": {"risk_level": "정상", "location": "주차장 A", "resolved": False},
        },
        {
            "id": "log_003",
            "text": "심야 창고 주변 배회. person:1. 주의.",
            "metadata": {"risk_level": "주의", "location": "창고 주변", "resolved": False},
        },
        {
            "id": "log_004",
            "text": "새벽 공장 외곽 침입. person:3 car:2. 위험.",
            "metadata": {"risk_level": "위험", "location": "공장 외곽", "resolved": False},
        },
        {
            "id": "log_005",
            "text": "낮 정문 정상 출입. person:5. 정상.",
            "metadata": {"risk_level": "정상", "location": "정문", "resolved": False},
        },
    ]

    collection.add(
        embeddings=[get_embedding(log["text"]) for log in initial_logs],
        documents=[log["text"] for log in initial_logs],
        ids=[log["id"] for log in initial_logs],
        metadatas=[log["metadata"] for log in initial_logs],
    )
    print(f"저장 완료. 총 {collection.count()}개\n")
    print("→ 프로그램을 종료하고 다시 실행하면 데이터가 그대로 유지됩니다.\n")
    
else :
    # select * from table; 와 같은 명령
    all_data = collection.get()
    for id_, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
        # resolved_mark = "✅" if meta["resolved"] else "⬜"
        print(f" {id_} | {meta['risk_level']} | {doc[:40]}")
    print()
    
# ─────────────────────────────────────────────────────────────
# STEP 2: UPDATE — 문서 내용 + 메타데이터 + 벡터 함께 변경
# 사건이 해결됐을 때 로그 전체를 업데이트하는 시나리오
#
# ⚠️ 핵심 주의사항:
# 문서 내용(documents)이 바뀌면 반드시 새 벡터(embeddings)도 함께 넘겨야 합니다.
# 벡터를 안 넘기면 내부 인덱스에는 예전 텍스트의 벡터가 그대로 남아서
# 새로 추가된 내용("경찰 출동 후 검거" 등)으로 유사도 검색이 안 됩니다.
# ─────────────────────────────────────────────────────────────
print("=== STEP 2: UPDATE (문서+벡터+메타 변경) ===")

before = collection.get(ids=["log_001"])
print(f"변경 전: {before['documents'][0]}")
# print(f"         resolved={before['metadatas'][0]['resolved']}")

new_text = "새벽 2시 창고 출입구 칩입. person:2 car:1. 위험. 경찰 출동 후 용의자 검거"
collection.update(
    ids=["log_001"],
    embeddings=[get_embedding(new_text)],   # 내용이 바뀌었으므로 백터 재생성 필수
    documents=[new_text],
    metadatas=[{"risk_level": "위험", "location": "창고 출입구", "resolved": True}]
)

after = collection.get(ids=["log_001"])
print(f"변경 후: {after['documents'][0]}")
print(f"         resolved={after['metadatas'][0]['resolved']}\n")