# GDS 그래프 분석 리포트

## 분석 기준

| 항목 | 값 |
| --- | ---: |
| GDS 그래프명 | `companyGraph` |
| 노드 수 | 8190 |
| 관계 수 | 51650 |

## PageRank 허브 Top 10

| 순위 | 노드 | 라벨 | PageRank |
| ---: | --- | --- | ---: |
| 1 | 서울 | Region | 643.423240 |
| 2 | 금융 | Section | 547.880434 |
| 3 | 제조 | Section | 197.856554 |
| 4 | 경기 | Region | 156.776375 |
| 5 | 서비스 | Section | 85.061345 |
| 6 | 부동산·건설 | Section | 79.064386 |
| 7 | IT·미디어 | Section | 75.149950 |
| 8 | 모름 | Section | 70.728101 |
| 9 | 유통·물류 | Section | 69.303498 |
| 10 | (주)신한금융지주회사 | ParentCompany | 59.080474 |

## Leiden 커뮤니티 Top 5

| 순위 | 커뮤니티 ID | 노드 수 | 주요 라벨 | 대표 노드 |
| ---: | ---: | ---: | --- | --- |
| 1 | 26 | 2951 | SubsidiaryCompany, ParentCompany, Region, Section | Kiwoom Asia Master Fund, Kiwoom Securities Holdings USA Inc., Kiwoom Securities USA Inc., 키움 시리우스사모투자합자회사, 키움뉴히어로4호스케일업펀드, 키움푸드테크 사모투자합자회사, 키움크리스제일호 사모투자합자회사, 키움뉴히어로5호디지털혁신펀드, 키움뉴히어로6호창업초기펀드, 케이에프비에스제이호기업재무안정사모투자 합자회사 |
| 2 | 20 | 1015 | SubsidiaryCompany, ParentCompany, Region, Section | 평화씨엠비(주), (주)평화이엔지, 창인인재개발원(주), (주)엠제이비전테크, (주)이노빌, 에이치디현대마린솔루션테크㈜, 에스제이지세움(주), 에스제이지아센텍(주), (주)에스제이지이브이(*5), 에스제이지중앙연구소(주) |
| 3 | 16 | 808 | SubsidiaryCompany, ParentCompany, Region, Section | KBI산업개발㈜, 에이치엘로지스앤코㈜, 목포신항만운영㈜, 배곧신도시지역특성화타운㈜, 에이치엘에코텍㈜, 다올칸피던스일반사모부동산 투자신탁제57호, 코람코전문투자형사모 부동산투자신탁117호(주1), 한일E&C㈜, 에스케이하이이엔지㈜, KOLON ENP EUROPE GmbH |
| 4 | 32 | 738 | SubsidiaryCompany, ParentCompany, Section, Region | 제주한림해상풍력(주), 탐라해상풍력발전(주), (주)디앤오, (주)미래엠, (주)디앤오씨엠, (주)디앤오리츠운용(*2), (주)엘지씨엔에스(*1), (주)지티이노비젼, (주)엘지스포츠, (주)엘지경영개발원 |
| 5 | 21 | 624 | SubsidiaryCompany, ParentCompany, Region, Section | 키움 AIM 일반사모부동산투자신탁제1호, 디앤디인베스트먼트㈜, ㈜디디아이오에스108 위탁관리부동산투자회사(주2), ㈜대유금형, 한국수력원자력(주), 한국남동발전(주), 한국중부발전(주), 한국서부발전(주), 한국남부발전(주), 한국전력기술(주) |

## 해석 기준

- PageRank는 중요한 노드와 연결된 노드를 더 높게 평가하는 중심성 지표입니다.
- Leiden 커뮤니티는 서로 촘촘히 연결된 노드 묶음을 찾는 커뮤니티 탐지 결과입니다.
- 이 리포트는 `ParentCompany`, `SubsidiaryCompany`, `Section`, `Region`과 기업 관계만 대상으로 하며, 뉴스 노드는 검색 근거용이라 제외했습니다.
