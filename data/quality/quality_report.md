# 트리플 품질 검증 리포트

## 요약

| 지표 | 값 |
| --- | ---: |
| 전체 트리플 수 | 26072 |
| 스키마 통과 트리플 수 | 26072 |
| 근거 원문 일치 트리플 수 | 17430 |
| 수동 검토 대상 수 | 17430 |
| 기각 트리플 수 | 8642 |
| 스키마 준수율 | 1.0000 |
| 근거 원문 일치율 | 0.6685 |
| 수동 검토 대상 비율 | 0.6685 |

## 계산 기준

- 스키마 준수율: 전체 추출 중 관계명과 source/target 타입이 온톨로지를 지킨 비율
- 근거 원문 일치율: 전체 추출 중 `evidence` 문자열이 원천 행에 실제 포함된 비율
  (단, `source_case="section_insert_v3"`는 원천 행 인용이 아니라 여러 Industry 관계를
  집계한 산업명 목록이므로, 원문 대조 대신 산업명이 동일한 키워드 규칙으로 해당
  Section을 실제로 도출하는지(`section_keyword_mismatch`) 검사한다.)
- 수동 검토 대상: 스키마 검사와 근거 일치 검사를 모두 통과한 트리플
- 샘플 정밀도: `manual_review_sample_50.csv`를 사람이 채점한 뒤 별도로 계산

## 주요 오류 유형

| 오류 | 건수 |
| --- | ---: |
| `missing_source_identifier` | 7407 |
| `section_keyword_mismatch` | 1235 |
| `source_row_not_found` | 380 |

## 수동 채점 안내

`manual_review_sample_50.csv`의 `manual_label`, `error_reason`, `fix_suggestion` 컬럼을 채우면 됩니다.

- `manual_label`: 정답 또는 오답
- `error_reason`: 오답일 때 오류 이유
- `fix_suggestion`: 수정 가능한 관계명, 방향, 근거 문장 등
