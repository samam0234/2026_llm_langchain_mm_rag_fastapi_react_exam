# ─────────────────────────────────────────────────────────────

# 예제 4 (실제 LLM 버전): ChatOpenAI + bind_tools Agent

# ─────────────────────────────────────────────────────────────

#

# 이 예제의 목표:

#   Mock Agent처럼 개발자가 직접 if문으로 Tool을 고르는 것이 아니라,

#   실제 LLM이 사용자의 자연어 질문을 읽고

#   어떤 Tool을 사용할지 스스로 판단하게 만드는 것입니다.

#

# 핵심 흐름:
#   1. 사용자가 질문한다.
#   2. LLM이 질문을 읽고 Tool 사용 여부를 판단한다.
#   3. Tool이 필요하면 AIMessage.tool_calls에 호출 정보가 담긴다.
#   4. 파이썬 코드가 실제 Tool 함수를 실행한다.
#   5. Tool 실행 결과를 ToolMessage로 다시 LLM에게 전달한다.
#   6. LLM이 Tool 결과를 읽고 최종 자연어 답변을 만든다.
#

# Mock 버전과 비교했을 때 달라진 점:

#  LLM의 tool_calls 판단으로 호출할 tool을 결정하는 부분

#  LLM의 최종 답변 생성으로 대체

# Tool 함수 자체는 그대로 재사용 가능

# ─────────────────────────────────────────────────────────────

import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# tool:

#   일반 Python 함수를 LangChain Tool로 바꿔주는 데코레이터입니다.

#   @tool을 붙이면 함수 이름, docstring, 파라미터 정보를

#   LLM이 이해할 수 있는 Tool 스키마로 변환합니다.
from langchain_core.tools import tool
from langchain_core.messages import (
  HumanMessage,
  ToolMessage,
  SystemMessage,

)

# .env 파일에서 OPENAI_API_KEY를 읽어옵니다.
load_dotenv()

