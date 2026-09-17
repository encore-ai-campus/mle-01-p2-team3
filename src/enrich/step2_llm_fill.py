"""2단계: 네이버 검색 MCP + LLM 에이전트로 남은 결측을 채운다.

1단계가 만든 data/work/todo_*.jsonl 을 한 건씩 에이전트에게 넘기고,
에이전트는 naver_search_mcp 의 검색 도구로 근거를 찾아 JSON 한 덩어리로 답한다.
결과는 근거 URL·신뢰도와 함께 data/work/filled_*.jsonl 에 누적 저장된다.

  # 키 확인 후 20건만 시험
  uv run python src/enrich/step2_llm_fill.py --target corp --limit 20
  # 전체 실행 (이미 처리한 건은 자동으로 건너뜀)
  uv run python src/enrich/step2_llm_fill.py --target all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enrich.common import (  # noqa: E402
    FILLED_CORP, FILLED_OV, FILLED_SB, FILLED_SUBS, REGIONS,
    TODO_CORP, TODO_OV, TODO_SB, TODO_SUBS, append_jsonl, read_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MCP_CONFIG = Path(__file__).resolve().parent / "mcp_servers.json"

SYSTEM_PROMPT = f"""너는 한국 기업 정보를 웹검색으로 확인해 채워 넣는 데이터 보강 담당자다.

사용 가능한 도구:
- naver_local_search: 회사명으로 업체 검색. 도로명주소와 업종 카테고리가 나온다. 가장 먼저 써라.
- naver_web_search: 웹문서 검색. "회사명 본사 주소", "회사명 업종" 처럼 구체적으로 검색해라.
- naver_news_search: 뉴스 검색. 위 둘로 부족할 때만.
- fetch_page_text: 검색 결과 링크의 본문을 직접 읽는다. 요약만으로 확신이 안 설 때만.

작업 규칙:
1. 도구 호출은 최대 4회. 4회 안에 못 찾으면 못 찾았다고 답한다.
2. 검색 결과에 근거가 있는 값만 적는다. 회사명으로 추측하지 마라.
   예를 들어 사명에 '건설'이 들어간다고 업종을 건설업으로 단정하면 안 된다.
3. 동명이인 주의. 검색 결과의 회사가 요청받은 회사와 같은 회사인지 확인해라.
   법인등록번호(crno)가 주어졌다면 그 번호가 보이는 문서를 우선한다.
4. 근거 URL을 제시할 수 없으면 status 를 "not_found" 로 답한다. 빈손이 틀린 값보다 낫다.
5. region 은 반드시 다음 중 하나이거나 빈 문자열이다: {", ".join(REGIONS)}
   해외 소재면 region 은 빈 문자열로 두고 domestic 을 "해외" 로 적는다.

답변 형식: 설명 없이 JSON 객체 하나만 출력한다. 코드펜스도 쓰지 마라.
"""

CORP_TASK = """다음 법인의 결측 항목을 채워라.

회사명: {corpNm}
법인등록번호: {crno}
이미 알고 있는 주소: {known_addr}
이미 알고 있는 업종: {known_sicNm}
채워야 할 항목: {need}

출력 JSON 키:
  "status": "found" 또는 "not_found"
  "addr": 본사 도로명주소 전체 (모르면 "")
  "region": 시·도 축약명 (모르면 "")
  "sicNm": 업종명. 한국표준산업분류 소분류 수준의 짧은 명사구. 예: "아파트 건설업", "소프트웨어 개발 및 공급업" (모르면 "")
  "evidence_url": 근거 페이지 URL
  "evidence": 근거가 된 문장 또는 검색 결과 한 줄 (100자 이내)
  "confidence": "high" / "medium" / "low"
"""

SUBS_TASK = """다음 회사의 결측 항목을 채워라. 이 회사는 '{parent_corpNm}' 의 연결대상 종속기업이다.

회사명: {subsidiary_name}
이미 알고 있는 주소: {known_addr}
이미 알고 있는 주요사업: {known_bizCtt}
채워야 할 항목: {need}

참고: 주소가 "대한민국" 처럼 뭉뚱그려진 값이면 실제 소재지를 찾아야 한다.
사명이 '...제일차', '...제이차' 로 끝나면 대개 자산유동화 특수목적회사(SPC)라 공개 정보가 거의 없다.
이런 경우 무리하게 채우지 말고 not_found 로 답해라.

출력 JSON 키:
  "status": "found" 또는 "not_found"
  "region": 시·도 축약명 (모르면 "")
  "addr": 소재지 주소 (모르면 "")
  "bizCtt": 주요 사업 내용을 짧은 명사구로 (모르면 "")
  "domestic": "국내" 또는 "해외" (모르면 "")
  "evidence_url": 근거 페이지 URL
  "evidence": 근거가 된 문장 (100자 이내)
  "confidence": "high" / "medium" / "low"
"""

_JSON_PAT = re.compile(r"\{.*\}", re.DOTALL)
ALLOWED_CONFIDENCE = {"high", "medium", "low"}


def _text_of(message) -> str:
    """responses/v1 출력은 content 가 블록 리스트일 수 있어 텍스트만 뽑는다."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text", "output_text"):
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return str(content)


def parse_answer(text: str) -> dict:
    """모델 출력에서 JSON 객체를 꺼내 값 범위를 검사한다."""
    match = _JSON_PAT.search(text)
    if not match:
        return {"status": "parse_error", "raw": text[:500]}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {"status": "parse_error", "raw": text[:500], "error": str(exc)}

    # 허용 범위를 벗어난 값은 버린다 (환각 방지 장치)
    if data.get("region") and data["region"] not in REGIONS:
        data["region_rejected"] = data.pop("region")
        data["region"] = ""
    if data.get("domestic") not in ("국내", "해외", "", None):
        data["domestic"] = ""
    if data.get("confidence") not in ALLOWED_CONFIDENCE:
        data["confidence"] = "low"
    if not str(data.get("evidence_url", "")).startswith("http"):
        data["evidence_url"] = ""
        data["status"] = "not_found"
    return data


def _usage_of(messages) -> dict[str, int]:
    """한 건을 처리하며 쓴 토큰을 합산한다 (비용 가늠용)."""
    usage = {"input_tokens": 0, "output_tokens": 0}
    for message in messages:
        meta = getattr(message, "usage_metadata", None) or {}
        usage["input_tokens"] += int(meta.get("input_tokens", 0) or 0)
        usage["output_tokens"] += int(meta.get("output_tokens", 0) or 0)
    return usage


async def run_one(agent, task_text: str, recursion_limit: int) -> tuple[dict, dict]:
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": task_text}]},
        config={"recursion_limit": recursion_limit},
    )
    messages = result["messages"]
    return parse_answer(_text_of(messages[-1])), _usage_of(messages)


def _todo_name(todo: dict) -> str:
    return str(todo.get("corpNm") or todo.get("subsidiary_name") or "")


def filter_todos(todos: list[dict], name_contains: list[str]) -> list[dict]:
    if not name_contains:
        return todos
    needles = [name for name in name_contains if name]
    return [
        todo for todo in todos
        if any(needle in _todo_name(todo) for needle in needles)
    ]


def done_keys_for(path: Path, retry_statuses: set[str]) -> set[str]:
    latest: dict[str, dict] = {}
    for row in read_jsonl(path):
        key = row.get("key")
        if not key:
            continue
        latest[key] = row

    done_keys: set[str] = set()
    for key, row in latest.items():
        result = row.get("result", {})
        status = result.get("status", "")
        confidence = result.get("confidence", "")
        if status in retry_statuses or f"confidence:{confidence}" in retry_statuses:
            continue
        done_keys.add(key)
    return done_keys


async def process(agent, todos: list[dict], out_path: Path, template: str,
                  concurrency: int, recursion_limit: int, label: str,
                  retry_statuses: set[str]) -> None:
    done_keys = done_keys_for(out_path, retry_statuses)
    pending = [todo for todo in todos if todo["key"] not in done_keys]
    print(f"[{label}] 전체 {len(todos):,}건 / 처리완료 {len(done_keys):,}건 / 이번 실행 {len(pending):,}건")
    if not pending:
        return

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    counter = {"done": 0, "found": 0, "error": 0, "in_tok": 0, "out_tok": 0}
    started = time.time()

    async def worker(todo: dict) -> None:
        fields = dict(todo)
        fields["need"] = ", ".join(todo["need"])
        usage: dict[str, int] = {}
        async with semaphore:
            try:
                answer, usage = await run_one(agent, template.format(**fields), recursion_limit)
            except Exception as exc:  # noqa: BLE001 - 한 건 실패로 배치를 멈추지 않는다
                answer = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        record = {**todo, "result": answer, "usage": usage}
        async with write_lock:
            append_jsonl(out_path, record)
            counter["done"] += 1
            counter["in_tok"] += usage.get("input_tokens", 0)
            counter["out_tok"] += usage.get("output_tokens", 0)
            if answer.get("status") == "found":
                counter["found"] += 1
            elif answer.get("status") in ("error", "parse_error"):
                counter["error"] += 1
            if counter["done"] % 10 == 0 or counter["done"] == len(pending):
                elapsed = time.time() - started
                print(f"  {counter['done']:,}/{len(pending):,} "
                      f"(found {counter['found']:,} / error {counter['error']:,}) "
                      f"{elapsed:,.0f}초 "
                      f"토큰 in {counter['in_tok']:,} / out {counter['out_tok']:,}")

    await asyncio.gather(*(worker(todo) for todo in pending))
    done = max(counter["done"], 1)
    print(f"[{label}] 저장: {out_path.name} | "
          f"건당 평균 토큰 in {counter['in_tok'] / done:,.0f} / out {counter['out_tok'] / done:,.0f}")


async def main_async(args: argparse.Namespace) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY 가 .env 에 없습니다.")
    if not os.getenv("NAVER_CLIENT_ID") or not os.getenv("NAVER_CLIENT_SECRET"):
        raise SystemExit("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 가 .env 에 없습니다.")

    from langchain.agents import create_agent
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_openai import ChatOpenAI

    config = json.loads(MCP_CONFIG.read_text(encoding="utf-8"))
    for server in config.values():
        # MCP 서버를 프로젝트 루트 기준으로 띄운다.
        server.setdefault("cwd", str(ROOT))
        server["command"] = server.get("command") or sys.executable
        if server["command"] == "python":
            server["command"] = sys.executable

    client = MultiServerMCPClient(config)
    tools = await client.get_tools()
    print(f"MCP 도구 {len(tools)}개 연결: {', '.join(t.name for t in tools)}")

    model = ChatOpenAI(model=args.model, use_responses_api=True, output_version="responses/v1")
    agent = create_agent(model, tools, system_prompt=SYSTEM_PROMPT)

    jobs = []
    if args.target in ("corp", "all"):
        jobs.append((TODO_CORP, FILLED_CORP, CORP_TASK, "법인"))
    if args.target in ("subs", "all"):
        jobs.append((TODO_SUBS, FILLED_SUBS, SUBS_TASK, "종속기업"))
    if args.target in ("ov", "all"):
        jobs.append((TODO_OV, FILLED_OV, CORP_TASK, "기업개요_최종"))
    if args.target in ("sb", "all"):
        jobs.append((TODO_SB, FILLED_SB, SUBS_TASK, "종속기업_정리"))

    for todo_path, out_path, template, label in jobs:
        todos = filter_todos(list(read_jsonl(todo_path)), args.name_contains)
        if args.limit:
            todos = todos[: args.limit]
        await process(agent, todos, out_path, template,
                      args.concurrency, args.recursion_limit, label,
                      set(args.retry_status))


def main() -> None:
    parser = argparse.ArgumentParser(description="웹검색 MCP + LLM 으로 기업 정보 결측 보강")
    parser.add_argument("--target",
                        choices=["corp", "subs", "ov", "sb", "all"], default="corp",
                        help="corp=통합CSV 법인, subs=통합CSV 종속기업, "
                             "ov=기업개요_최종.csv, sb=종속기업_정리.csv")
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N건만 처리 (0=전체)")
    parser.add_argument("--concurrency", type=int, default=4, help="동시 처리 건수")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"))
    parser.add_argument("--recursion-limit", type=int, default=12,
                        help="한 건당 에이전트 최대 스텝 수")
    parser.add_argument("--name-contains", action="append", default=[],
                        help="회사명에 이 문자열이 들어간 작업만 처리한다. 여러 번 지정 가능")
    parser.add_argument("--retry-status", action="append", default=[],
                        help="기존 결과가 이 status 이면 다시 처리한다. 예: error, not_found, parse_error. "
                             "confidence:low 도 가능")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
