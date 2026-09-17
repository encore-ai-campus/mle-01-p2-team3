import json
import os
import pickle
from email.utils import parsedate_to_datetime
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from langchain_openai import OpenAIEmbeddings


# 뉴스 기사를 Aura 에 News 노드로 올리고, ParentCompany 와 RELATED_TO 로 잇고,
# 임베딩을 붙여 벡터 인덱스(news_vec)까지 만든다.
# 기업 그래프(ParentCompany 등)는 이미 Aura 에 올라가 있어야 관계가 생긴다.

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

AURA_URI = os.getenv("AURA_URI")
AURA_USER = os.getenv("AURA_USER", "neo4j")
AURA_PASSWORD = os.getenv("AURA_PASSWORD")

if not all([AURA_URI, AURA_PASSWORD]):
    raise RuntimeError(
        "AURA_URI, AURA_USER, AURA_PASSWORD를 .env에서 확인하세요."
    )

NEWS_FILE = PROJECT_ROOT / "data" / "clean" / "최종_뉴스기사.jsonl"

# 한 번 계산한 임베딩은 파일에 저장해 두고 다시 쓴다
EMB_CACHE_FILE = PROJECT_ROOT / "data" / "news_emb_cache.pkl"

# 검색 도구(src/agent/tools/news_vector.py)와 같은 값이어야 한다
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 768
VECTOR_INDEX = "news_vec"

BATCH_SIZE = 1000


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
                driver = GraphDatabase.driver(
                    candidate_uri,
                    auth=(candidate_user, password),
                )
                driver.verify_connectivity()
                return driver
            except Exception as error:
                last_error = error
    raise last_error


def to_date(value):
    """'Thu, 17 Sep 2026 13:36:00 +0900' -> '2026-09-17'. 못 읽으면 빈 문자열."""
    try:
        return parsedate_to_datetime(value).date().isoformat()
    except (TypeError, ValueError):
        return ""


def read_news():
    with NEWS_FILE.open(encoding="utf-8") as file:
        articles = [json.loads(line) for line in file if line.strip()]

    rows = [
        {
            "id": article["record_id"],
            "crno": article["company_crno"],
            "date": to_date(article.get("published_at")),
            "title": (article.get("title") or "").strip(),
            "summary": (article.get("description") or "").strip(),
            "publisher": (article.get("publisher") or "").strip(),
            "url": (article.get("original_url") or "").strip(),
        }
        for article in articles
    ]

    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("record_id 중복이 있습니다.")

    return rows


def embed_texts(texts):
    """문서 리스트 -> 768차원 임베딩 리스트. 캐시에 없는 것만 OpenAI 로 계산한다."""
    cache = (
        pickle.loads(EMB_CACHE_FILE.read_bytes())
        if EMB_CACHE_FILE.exists()
        else {}
    )

    new_texts = list(dict.fromkeys(text for text in texts if text not in cache))

    if new_texts:
        embedder = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=EMBED_DIM)
        cache.update(zip(new_texts, embedder.embed_documents(new_texts)))
        EMB_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        EMB_CACHE_FILE.write_bytes(pickle.dumps(cache))

    unique_count = len(set(texts))
    print(f"임베딩: 서로 다른 문장 {unique_count:,}건 중 새로 계산 {len(new_texts):,}건")

    return [cache[text] for text in texts]


def run_in_batches(session, query, rows):
    for start in range(0, len(rows), BATCH_SIZE):
        session.run(query, rows=rows[start:start + BATCH_SIZE]).consume()


def check_company_crnos(session, rows):
    """기사의 crno 가 그래프에 있는지 확인한다. 없는 crno 는 관계가 안 생긴다."""
    graph_crnos = {
        record["crno"]
        for record in session.run("MATCH (p:ParentCompany) RETURN p.crno AS crno")
    }
    news_crnos = {row["crno"] for row in rows}
    missing = sorted(news_crnos - graph_crnos)

    print(f"기사에 나오는 기업 {len(news_crnos):,}곳 / 그래프에 없음 {len(missing):,}곳")

    if missing:
        print("  ->", missing[:10])


def create_constraint(session):
    session.run(
        """
        CREATE CONSTRAINT news_id_unique IF NOT EXISTS
        FOR (n:News) REQUIRE n.id IS UNIQUE
        """
    ).consume()


def load_news_nodes(session, rows):
    run_in_batches(
        session,
        """
        UNWIND $rows AS row
        MERGE (n:News {id: row.id})
        SET n.crno = row.crno,
            n.date = row.date,
            n.title = row.title,
            n.summary = row.summary,
            n.publisher = row.publisher,
            n.url = row.url
        """,
        rows,
    )


def load_related_to(session, rows):
    run_in_batches(
        session,
        """
        UNWIND $rows AS row
        MATCH (n:News {id: row.id})
        MATCH (p:ParentCompany {crno: row.crno})
        MERGE (n)-[:RELATED_TO]->(p)
        """,
        rows,
    )


def load_embeddings(session, rows):
    # 제목 + 요약을 이어 임베딩한다
    texts = [f"{row['title']} {row['summary']}".strip() for row in rows]
    vectors = embed_texts(texts)

    run_in_batches(
        session,
        """
        UNWIND $rows AS row
        MATCH (n:News {id: row.id})
        CALL db.create.setNodeVectorProperty(n, 'embedding', row.vec)
        """,
        [
            {"id": row["id"], "vec": vector}
            for row, vector in zip(rows, vectors)
        ],
    )


def create_vector_index(session):
    session.run(
        f"""
        CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS
        FOR (n:News) ON n.embedding
        OPTIONS {{indexConfig: {{
          `vector.dimensions`: {EMBED_DIM},
          `vector.similarity_function`: 'cosine'
        }}}}
        """
    ).consume()

    session.run("CALL db.awaitIndexes()").consume()


def print_results(session):
    counts = session.run(
        """
        MATCH (n:News)
        RETURN count(n) AS news,
               count(n.embedding) AS embedded,
               count { MATCH (:News)-[:RELATED_TO]->(:ParentCompany) } AS related_to,
               count { MATCH (m:News) WHERE NOT (m)-[:RELATED_TO]->() } AS unlinked
        """
    ).single()

    print("\n[뉴스 적재 결과]")
    print(counts.data())

    print("\n[벡터 인덱스]")
    for record in session.run(
        """
        SHOW VECTOR INDEXES YIELD name, state, labelsOrTypes, properties
        RETURN name, state, labelsOrTypes, properties
        """
    ):
        print(record.data())


def main():
    rows = read_news()
    print(f"기사 {len(rows):,}건 읽음")

    driver = connect_aura(AURA_URI, AURA_USER, AURA_PASSWORD)

    try:
        print("Aura 연결 성공")

        with driver.session() as session:
            check_company_crnos(session, rows)

            create_constraint(session)
            print("제약조건 생성 확인")

            load_news_nodes(session, rows)
            print("News 노드 적재 완료")

            load_related_to(session, rows)
            print("RELATED_TO 관계 적재 완료")

            load_embeddings(session, rows)
            print("임베딩 적재 완료")

            create_vector_index(session)
            print("벡터 인덱스 생성 확인")

            print_results(session)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
