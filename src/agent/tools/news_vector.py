import os

from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from langchain_openai import OpenAIEmbeddings

# ====== 환경변수 ======

# 뉴스 노드와 벡터 인덱스(news_vec)는 로컬이 아니라 Aura 에 올라가 있다.
# 그래서 text2cypher 의 driver(NEO4J_*) 와 별개로 AURA_* 로 연결한다.
ENV_PATH = Path(__file__).resolve().parents[3] / ".env"

load_dotenv(ENV_PATH)

AURA_URI = os.getenv("AURA_URI")
AURA_USER = os.getenv("AURA_USER", "neo4j")
AURA_PASSWORD = os.getenv("AURA_PASSWORD")

# 적재할 때와 같은 모델/차원이어야 인덱스와 비교가 된다
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 768
VECTOR_INDEX = "news_vec"

MAX_TOP_K = 10


# ====== Aura 연결 (첫 호출 때 한 번만) ======

# import 시점에 붙으면 Aura 가 꺼져 있을 때 에이전트 전체가 안 뜬다.
# 그래서 도구가 실제로 불릴 때 연결하고, 실패하면 도구 결과로 에러를 돌려준다.
_aura_driver = None
_embedder = None


def connect_aura(uri, user, password):
    """Aura 에 붙는다. 인증서/사용자명 문제로 실패하면 ssc 주소와 인스턴스 ID 로 다시 시도한다."""
    host = uri.split("://")[-1]
    instance_id = host.split(".")[0]

    uris = [uri]
    if not uri.startswith("neo4j+ssc"):
        uris.append(f"neo4j+ssc://{host}")
    users = list(dict.fromkeys([user, instance_id]))

    last_error = None
    for candidate_uri in uris:
        for candidate_user in users:
            try:
                new_driver = GraphDatabase.driver(
                    candidate_uri,
                    auth=(candidate_user, password),
                )
                new_driver.verify_connectivity()
                return new_driver
            except Exception as error:
                last_error = error
    raise last_error


def get_aura_driver():
    global _aura_driver

    if _aura_driver is None:
        if not AURA_URI or not AURA_PASSWORD:
            raise RuntimeError(
                "Aura 환경변수가 설정되지 않았습니다. "
                "AURA_URI, AURA_USER, AURA_PASSWORD를 확인하세요."
            )
        _aura_driver = connect_aura(AURA_URI, AURA_USER, AURA_PASSWORD)

    return _aura_driver


def get_embedder():
    global _embedder

    if _embedder is None:
        _embedder = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIM)

    return _embedder


# ====== TOOL3 ======

def search_news(question: str, top_k: int = 3):
    top_k = max(1, min(int(top_k), MAX_TOP_K))

    try:
        driver = get_aura_driver()
        query_vector = get_embedder().embed_query(question)

        # Aura 5.x 라 SEARCH 절 대신 queryNodes 프로시저를 쓴다
        records, summary, keys = driver.execute_query(
            """
            CALL db.index.vector.queryNodes($index, $k, $q)
            YIELD node, score
            OPTIONAL MATCH (node)-[:RELATED_TO]->(p:ParentCompany)
            RETURN node.title     AS title,
                   node.summary   AS summary,
                   node.date      AS date,
                   node.publisher AS publisher,
                   node.url       AS url,
                   p.name         AS company,
                   p.crno         AS crno,
                   score
            ORDER BY score DESC
            """,
            index=VECTOR_INDEX,
            k=top_k,
            q=query_vector,
        )

    except Exception as error:
        return {
            "question": question,
            "results": [],
            "error": f"뉴스 벡터 검색 실패: {type(error).__name__}: {error}",
        }

    # score = (1 + 코사인) / 2 라 관련 없는 문장도 0.5 근처가 나온다. 원래 코사인도 같이 준다.
    results = [
        {
            "title": record["title"],
            "summary": record["summary"],
            "date": record["date"],
            "publisher": record["publisher"],
            "url": record["url"],
            "company": record["company"],
            "crno": record["crno"],
            "score": round(record["score"], 4),
            "cosine": round(2 * record["score"] - 1, 4),
        }
        for record in records
    ]

    return {
        "question": question,
        "found": len(results) > 0,
        "results": results,
    }


#도구3 독스트링
search_news.__doc__ = f"""
기업 관련 뉴스 기사를 의미(벡터) 검색하여
질문과 가장 가까운 기사와 원문 URL을 반환합니다.

question에는 찾고 싶은 뉴스의 내용을 자연어로 넣으세요.
기업명이 있다면 함께 넣으면 더 정확합니다.
예: "삼성전자 반도체 투자 확대", "현대자동차 전기차 공장"

top_k는 가져올 기사 수입니다. (기본 3, 최대 {MAX_TOP_K})

반환되는 results의 각 항목:

- title: 기사 제목
- summary: 기사 요약
- date: 기사 날짜 (YYYY-MM-DD)
- publisher: 언론사
- url: 기사 원문 URL
- company: 기사와 연결된 모기업명 (ParentCompany)
- crno: 모기업 법인등록번호
- score: 벡터 유사도 (0~1, 관련 없어도 0.5 근처가 나옵니다)
- cosine: 코사인 유사도 (-1~1)

다음과 같은 질문에 사용하세요.

- 특정 기업의 최근 뉴스, 소식, 동향
- 특정 주제(투자, 인수합병, 실적 등)와 관련된 기사
- 답변의 근거가 되는 기사 원문 링크가 필요한 경우

검색 결과는 의미상 가장 가까운 기사일 뿐
질문과 반드시 관련 있는 것은 아닙니다.
제목과 요약을 확인해 질문과 관련 없는 기사는 답변에서 제외하세요.


[요청 기업과 검색 결과 구분]

사용자가 특정 기업, 계열사, 종속기업의 뉴스를 요청한 경우
검색 결과가 실제로 그 기업의 뉴스인지 확인하세요.

요청한 기업의 직접적인 뉴스가 없고
모기업이나 다른 계열사의 뉴스만 검색된 경우,
그 뉴스를 요청한 기업의 뉴스인 것처럼 답하지 마세요.

관련 기업의 뉴스를 대신 제시할 수는 있지만,
먼저 요청한 기업의 직접적인 뉴스는 확인되지 않았다고 설명하고
"대신 모기업 OOO의 관련 뉴스입니다."처럼
대체된 대상이 무엇인지 명확히 밝혀주세요.

대체할 만한 관련 뉴스도 없다면
현재 검색 결과에서는 요청한 기업의 뉴스를
확인할 수 없다고 답하세요.

여러 기업의 뉴스를 요청받은 경우
어떤 기업들을 실제로 검색했는지와
그중 직접 뉴스가 확인된 기업을 구분하세요.
검색하지 않은 기업을 "뉴스가 없는 기업"으로 표현하지 마세요.


기업 간 관계, 종속기업, 지역, 업종 조회에는
이 도구 대신 search_graph를 사용하세요.

답변에 기사를 제시할 때는 각 기사의 score를 유사도로 함께 표시하세요.
"""