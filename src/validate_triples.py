import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = [
    "subject",
    "subject_type",
    "relation",
    "object",
    "object_type",
    "evidence",
]

SCHEMA_ERROR_PREFIXES = (
    "missing_field:",
    "empty_field:",
    "unknown_relation:",
    "invalid_signature:",
    "invalid_json:",
)

QUOTE_ERROR_PREFIXES = (
    "empty_evidence",
    "missing_source_identifier",
    "source_row_not_found",
    "quote_not_found_in_source",
    "section_keyword_mismatch",
)

# section_insert_v3의 IN_INDUSTRY evidence도 회사명·주소·업종명을 포함한
# 원문 행 인용이다. source_row가 가리키는 원천 행에 evidence가 포함되는지
# 다른 관계와 동일한 방식으로 검증한다.
SECTION_INSERT_V3_CASE = "section_insert_v3"

SECTION_ROWS = [
    {"name": "금융", "category": ["SPC", "펀드", "지주", "은행", "증권", "보험", "금융", "투자", "유동화", "신탁", "대출", "자산운용", "캐피탈", "대부", "여신", "할부", "집합투자", "사모", "특수목적", "기업어음", "벤처", "조합", "손해사정"]},
    {"name": "제조", "category": ["전자", "자동차", "화학", "소재", "기계", "제조", "반도체", "전지", "케이블", "시멘트", "레미콘", "플라스틱", "철강", "제강", "금속", "섬유", "의류", "선박", "부품"]},
    {"name": "IT·미디어", "category": ["소프트웨어", "통신", "게임", "방송", "콘텐츠", "IT서비스", "정보통신", "정보기술", "컴퓨터", "데이터베이스", "시스템", "광고", "영화", "드라마", "음악", "엔터테인먼트", "매니지먼트", "연예", "미디어", "컨텐츠", "출판", "인터넷", "플랫폼"]},
    {"name": "부동산·건설", "category": ["부동산", "임대", "건설", "시공", "개발사업", "시행", "분양", "주택", "건축", "엔지니어링"]},
    {"name": "서비스", "category": ["호텔", "교육", "컨설팅", "연구", "정비", "서비스", "골프", "여행", "학원", "시설관리", "콜센터", "텔레마케팅", "고객센터", "사업지원", "경영", "경비", "청소", "인력", "전시회", "전시대행", "테마파크", "콘도"]},
    {"name": "유통·물류", "category": ["도소매", "무역", "운송", "창고", "유통", "물류", "도매", "소매", "도ㆍ소매", "운수", "항만", "해운", "택배", "백화점"]},
    {"name": "바이오·헬스케어", "category": ["제약", "의료", "화장품", "의약품", "바이오", "헬스", "병원", "건강"]},
    {"name": "에너지·환경", "category": ["발전", "태양광", "폐기물", "가스", "에너지", "전력", "전기업", "석유", "원유", "연료", "열공급", "증기", "환경"]},
    {"name": "식품·농업", "category": ["식품", "외식", "농축산", "음료", "음식", "구내식당", "급식", "정육", "농업", "축산", "양돈", "가금류", "작물", "수산", "어업", "생수", "주류", "사료"]},
    {"name": "모름", "category": []},
]


def classify_industry(industry_name: str) -> list[str]:
    normalized_name = (industry_name or "").casefold()
    matches = [
        section["name"]
        for section in SECTION_ROWS
        if section["name"] != "모름"
        and any(keyword.casefold() in normalized_name for keyword in section["category"])
    ]
    return matches or ["모름"]


def _section_evidence_ok(evidence: str, object_field: str) -> bool:
    section_name = object_field.split(":", 1)[1] if ":" in object_field else object_field
    industry_names = [part.strip() for part in evidence.split(",") if part.strip()]

    matched_sections: set[str] = set()
    for industry_name in industry_names:
        matched_sections.update(classify_industry(industry_name))
    matched_sections.discard("모름")
    if not matched_sections:
        matched_sections = {"모름"}

    return section_name in matched_sections


@dataclass
class ValidationResult:
    line_no: int | None
    triple: dict[str, Any]
    errors: list[str]

    @property
    def schema_ok(self) -> bool:
        return not any(error.startswith(SCHEMA_ERROR_PREFIXES) for error in self.errors)

    @property
    def quote_ok(self) -> bool:
        return not any(error.startswith(QUOTE_ERROR_PREFIXES) for error in self.errors)

    @property
    def is_valid(self) -> bool:
        # 교안 기준의 "검토 대상": 스키마와 근거 원문 일치 검사를 모두 통과한 행.
        return self.schema_ok and self.quote_ok

    @property
    def has_schema_error(self) -> bool:
        return not self.schema_ok

    @property
    def has_evidence_error(self) -> bool:
        return not self.quote_ok


def build_allowed_signatures(ontology: dict[str, Any]) -> dict[str, set[tuple[str, str]]]:
    allowed: dict[str, set[tuple[str, str]]] = {}
    for relation, spec in ontology.get("relationships", {}).items():
        signatures = spec.get("signatures")
        if signatures is None and "source" in spec and "target" in spec:
            signatures = [{"source": spec["source"], "target": spec["target"]}]

        allowed[relation] = {
            (signature["source"], signature["target"])
            for signature in signatures or []
            if signature.get("source") and signature.get("target")
        }
    return allowed


def validate_triple(
    triple: dict[str, Any],
    allowed_signatures: dict[str, set[tuple[str, str]]],
    *,
    line_no: int | None = None,
    source_text: str | None = None,
) -> ValidationResult:
    triple = dict(triple)
    errors: list[str] = []

    _ensure_source_doc_id(triple)

    for field in REQUIRED_FIELDS:
        if field not in triple:
            errors.append(f"missing_field:{field}")
        elif _is_blank(triple[field]):
            errors.append(f"empty_field:{field}")

    relation = str(triple.get("relation", "")).strip()
    subject_type = str(triple.get("subject_type", "")).strip()
    object_type = str(triple.get("object_type", "")).strip()

    if relation:
        if relation not in allowed_signatures:
            errors.append(f"unknown_relation:{relation}")
        elif (subject_type, object_type) not in allowed_signatures[relation]:
            errors.append(f"invalid_signature:{relation}:{subject_type}->{object_type}")

    evidence = str(triple.get("evidence", "")).strip()
    if not evidence:
        errors.append("empty_evidence")

    if "source_doc_id" not in triple:
        errors.append("missing_source_identifier")
    elif source_text is not None and evidence and not _quote_in_source(evidence, source_text):
        errors.append("quote_not_found_in_source")

    return ValidationResult(line_no=line_no, triple=triple, errors=errors)


def summarize_results(results: list[ValidationResult]) -> dict[str, Any]:
    total = len(results)
    schema_pass = sum(result.schema_ok for result in results)
    quote_pass = sum(result.quote_ok for result in results)
    review_population = sum(result.is_valid for result in results)

    error_counts: dict[str, int] = {}
    relation_counts: dict[str, int] = {}
    for result in results:
        relation = str(result.triple.get("relation", ""))
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
        for error in result.errors:
            error_counts[error] = error_counts.get(error, 0) + 1

    return {
        "total_triples": total,
        "schema_pass_triples": schema_pass,
        "quote_pass_triples": quote_pass,
        "review_population_size": review_population,
        "rejected_triples": total - review_population,
        "schema_compliance_rate": _rate(schema_pass, total),
        "quote_match_rate": _rate(quote_pass, total),
        "review_population_rate": _rate(review_population, total),
        "error_counts": dict(sorted(error_counts.items(), key=lambda item: (-item[1], item[0]))),
        "relation_counts": dict(sorted(relation_counts.items())),
    }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                row = {
                    "_json_error": str(exc),
                    "_raw_line": line,
                }
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def load_source_table(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}

    table: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row_no, row in enumerate(reader):
            table[row_no] = " ".join(str(value) for value in row.values() if value not in (None, ""))
    return table


def validate_file(
    triples_path: Path,
    ontology_path: Path,
    *,
    source_table_path: Path | None = None,
) -> list[ValidationResult]:
    allowed_signatures = build_allowed_signatures(load_json(ontology_path))
    triples = load_jsonl(triples_path)
    source_table = load_source_table(source_table_path)

    results: list[ValidationResult] = []
    for triple in triples:
        line_no = triple.pop("_line_no", None)
        if "_json_error" in triple:
            results.append(
                ValidationResult(
                    line_no=line_no,
                    triple=triple,
                    errors=[f"invalid_json:{triple['_json_error']}"],
                )
            )
            continue

        source_text = None
        if source_table:
            source_row = _to_int(triple.get("source_row"))
            if source_row is None or source_row not in source_table:
                result = validate_triple(triple, allowed_signatures, line_no=line_no)
                result.errors.append("source_row_not_found")
                results.append(result)
                continue
            source_text = source_table[source_row]

        results.append(
            validate_triple(
                triple,
                allowed_signatures,
                line_no=line_no,
                source_text=source_text,
            )
        )
    return results


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manual_sample(path: Path, results: list[ValidationResult], sample_size: int, seed: int) -> None:
    review_population = [result for result in results if result.is_valid]
    sample = random.Random(seed).sample(review_population, min(sample_size, len(review_population)))

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "line_no",
        "subject",
        "subject_type",
        "relation",
        "object",
        "object_type",
        "evidence",
        "source_doc_id",
        "manual_label",
        "error_reason",
        "fix_suggestion",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in sample:
            row = {field: result.triple.get(field, "") for field in fieldnames}
            row["line_no"] = result.line_no
            row["manual_label"] = ""
            row["error_reason"] = ""
            row["fix_suggestion"] = ""
            writer.writerow(row)


def write_markdown_report(path: Path, summary: dict[str, Any], gold_metrics: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    error_lines = "\n".join(
        f"| `{error}` | {count} |" for error, count in summary["error_counts"].items()
    )
    if not error_lines:
        error_lines = "| 없음 | 0 |"

    gold_section = ""
    if gold_metrics is not None:
        gold_section = f"""
## 골드 완전일치

| 지표 | 값 |
| --- | ---: |
| TP | {gold_metrics["tp"]} |
| FP | {gold_metrics["fp"]} |
| FN | {gold_metrics["fn"]} |
| 정밀도 | {gold_metrics["precision"]:.4f} |
| 재현율 | {gold_metrics["recall"]:.4f} |
| F1 | {gold_metrics["f1"]:.4f} |
"""

    text = f"""# 트리플 품질 검증 리포트

## 요약

| 지표 | 값 |
| --- | ---: |
| 전체 트리플 수 | {summary["total_triples"]} |
| 스키마 통과 트리플 수 | {summary["schema_pass_triples"]} |
| 근거 원문 일치 트리플 수 | {summary["quote_pass_triples"]} |
| 수동 검토 대상 수 | {summary["review_population_size"]} |
| 기각 트리플 수 | {summary["rejected_triples"]} |
| 스키마 준수율 | {summary["schema_compliance_rate"]:.4f} |
| 근거 원문 일치율 | {summary["quote_match_rate"]:.4f} |
| 수동 검토 대상 비율 | {summary["review_population_rate"]:.4f} |

## 계산 기준

- 스키마 준수율: 전체 추출 중 관계명과 source/target 타입이 온톨로지를 지킨 비율
- 근거 원문 일치율: 전체 추출 중 `evidence` 문자열이 해당 `source_row`의 원천 행에
  실제 포함되는 비율. Section(`IN_INDUSTRY`) 관계도 같은 방식으로 검사한다.
- 수동 검토 대상: 스키마 검사와 근거 일치 검사를 모두 통과한 트리플
- 샘플 정밀도: `manual_review_sample_50.csv`를 사람이 채점한 뒤 별도로 계산

## 주요 오류 유형

| 오류 | 건수 |
| --- | ---: |
{error_lines}
{gold_section}
## 수동 채점 안내

`manual_review_sample_50.csv`의 `manual_label`, `error_reason`, `fix_suggestion` 컬럼을 채우면 됩니다.

- `manual_label`: 정답 또는 오답
- `error_reason`: 오답일 때 오류 이유
- `fix_suggestion`: 수정 가능한 관계명, 방향, 근거 문장 등
"""
    path.write_text(text, encoding="utf-8")


def compute_gold_exact_match(results: list[ValidationResult], gold_path: Path) -> dict[str, Any]:
    predictions = {_triple_key(result.triple) for result in results if result.is_valid}
    gold_rows = load_jsonl(gold_path)
    gold = {_triple_key(row) for row in gold_rows}

    tp = len(predictions & gold)
    fp = len(predictions - gold)
    fn = len(gold - predictions)
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_quality_check(
    triples_path: Path,
    ontology_path: Path,
    output_dir: Path,
    *,
    source_table_path: Path | None = None,
    gold_path: Path | None = None,
    sample_size: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    results = validate_file(
        triples_path,
        ontology_path,
        source_table_path=source_table_path,
    )
    summary = summarize_results(results)
    gold_metrics = compute_gold_exact_match(results, gold_path) if gold_path else None
    if gold_metrics is not None:
        summary["gold_exact_match"] = gold_metrics

    schema_pass = [result.triple for result in results if result.schema_ok]
    quote_pass = [result.triple for result in results if result.quote_ok]
    review_population = [result.triple for result in results if result.is_valid]
    rejected = [
        {
            "line_no": result.line_no,
            "errors": result.errors,
            **result.triple,
        }
        for result in results
        if not result.is_valid
    ]

    write_jsonl(output_dir / "schema_pass_triples.jsonl", schema_pass)
    write_jsonl(output_dir / "quote_pass_triples.jsonl", quote_pass)
    write_jsonl(output_dir / "approved_triples.jsonl", review_population)
    write_jsonl(output_dir / "rejected_triples.jsonl", rejected)
    write_manual_sample(output_dir / "manual_review_sample_50.csv", results, sample_size, seed)
    write_markdown_report(output_dir / "quality_report.md", summary, gold_metrics)
    (output_dir / "quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate knowledge graph triples.")
    parser.add_argument("--triples", type=Path, default=Path("data/clean/최종_기업관계_트리플.jsonl"))
    parser.add_argument("--ontology", type=Path, default=Path("src/agent/tools/graph_ontology.json"))
    parser.add_argument("--source-table", type=Path, default=Path("data/clean/최종_모기업_계열사_종속기업_통합.csv"))
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/quality"))
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_quality_check(
        args.triples,
        args.ontology,
        args.output_dir,
        source_table_path=args.source_table,
        gold_path=args.gold,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _ensure_source_doc_id(triple: dict[str, Any]) -> None:
    if not _is_blank(triple.get("source_doc_id")):
        return
    source_row = triple.get("source_row")
    if _is_blank(source_row):
        return

    source_case = triple.get("source_case")
    if _is_blank(source_case):
        triple["source_doc_id"] = f"row:{source_row}"
    else:
        triple["source_doc_id"] = f"case:{source_case}:row:{source_row}"


def _is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def _quote_in_source(evidence: str, source_text: str) -> bool:
    return _normalize_text(evidence) in _normalize_text(source_text)


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _triple_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("subject", "")).strip(),
        str(row.get("relation", "")).strip(),
        str(row.get("object", "")).strip(),
    )


if __name__ == "__main__":
    main()
