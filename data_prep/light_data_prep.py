"""
LIGHT 데이터셋 기반 King 중심 GraphRAG 전처리 파이프라인.

폴더 구조
---------
제 프로젝트 루트 폴더를 기준으로 사용합니다.

project_root/
├─ data
│   ├─ light_data.pkl
│   ├─ light_unseen_data.pkl
│   └─ light_environment.pkl
└─ data_prep
    ├─ light_data_prep.py
    ├─ light_data_prep.ipynb
    ├─ light_data_prep_persona.py
    ├─ light_data_prep_persona.ipynb
    └─ OUTPUT/
        ├─ metrics/
        ├─ subsets/
        ├─ neo4j/
        ├─ graphrag/
        └─ preprocessing_summary.json


중요한 설계 원칙
---------------
1. node와 edge를 같은 실행 안에서 함께 생성합니다.
2. character_id, location_id, object_id 생성 함수를 node/edge에서 공통으로 사용합니다.
3. 특히 object_id는 hash suffix를 붙여 ID 충돌을 방지합니다.
4. 최종 검증에서 edge가 참조하는 node ID가 실제 node CSV에 존재하는지 확인합니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


# =============================================================================
# 0. 프로젝트 기본 경로
# =============================================================================

# 이 py 파일이 있는 폴더를 프로젝트 루트로 봅니다.
# 사용자의 폴더 구조가 다음과 같다는 전제입니다.
#   프로젝트루트/raw/*.pkl
#   프로젝트루트/OUTPUT/
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = SCRIPT_DIR.parent / "data"
DEFAULT_OUT_DIR = SCRIPT_DIR / "OUTPUT"


# =============================================================================
# 1. 전역 설정값
# =============================================================================

# 이번 프로젝트에서 중심으로 삼는 character입니다.
FOCUS_CHARACTER = "king"

# 일반명사 성격이 강하지만 LIGHT에서는 실제 character로 쓰일 수 있으므로 제거하지 않고 flag만 달아 둡니다.
GENERIC_NAMES = {
    "person", "people", "man", "woman", "child", "boy", "girl", "guard",
    "servant", "peasant", "villager", "animal", "horse", "bird", "rat",
    "fish", "family", "wife", "husband", "lord", "lady", "guest", "visitor",
    "father", "mother", "friend", "stranger", "traveler", "farmer",
}

# pronoun, quantifier처럼 Character node로 만들면 안 되는 단어입니다.
EXCLUDED_CHARACTER_NAMES = {
    "i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his",
    "she", "her", "hers", "we", "us", "our", "they", "them", "their",
    "it", "its", "one", "some", "many", "other", "another", "all", "few",
    "none", "someone", "something", "anyone", "anything", "everyone",
}

# 자주 나오지만 서사적 의미가 약할 수 있는 배경 object입니다. 제거하지 않고 flag만 남깁니다.
BACKGROUND_OBJECTS = {
    "table", "wall", "chair", "window", "door", "floor", "room", "candle",
    "bed", "bench", "shelf", "shelves", "stool", "rug", "carpet", "curtain",
    "curtains", "ceiling", "fireplace", "torch", "lamp", "lantern",
}

# 왕/궁정/모험 서사에서 의미가 있을 가능성이 높은 object keyword입니다.
STORY_RELEVANT_OBJECT_KEYWORDS = {
    "sword", "crown", "gold", "coin", "coins", "throne", "letter", "scroll",
    "book", "key", "ring", "jewel", "jewels", "gem", "gems", "armor",
    "shield", "dagger", "knife", "potion", "chest", "treasure", "map",
}


# =============================================================================
# 2. 기본 유틸 함수
# =============================================================================


def normalize_name(name: Any) -> str:
    """엔티티 이름을 비교/집계하기 쉬운 형태로 정규화합니다.

    처리 내용:
    - None은 빈 문자열로 처리
    - 소문자화
    - 앞뒤 공백 제거
    - 앞쪽 관사 the/a/an 반복 제거
    - 특수문자 정리

    관사를 반복 제거하는 이유:
    LIGHT object 중에는 ``an a bar``처럼 관사가 중첩된 값이 있습니다.
    한 번만 제거하면 node 생성 시점과 id 생성 시점의 이름이 달라질 수 있으므로 반복 제거합니다.
    """
    if name is None:
        return ""

    value = str(name).strip().lower()

    # "an a bar" -> "a bar" -> "bar"처럼 앞쪽 관사를 더 이상 없어질 때까지 제거합니다.
    prev = None
    while prev != value:
        prev = value
        value = re.sub(r"^(the|a|an)\s+", "", value)

    # 영문자, 숫자, 공백, apostrophe, hyphen만 남기고 나머지는 공백으로 바꿉니다.
    value = re.sub(r"[^a-z0-9\s'\-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def display_name(name: str) -> str:
    """정규화된 이름을 보고서/CSV에서 보기 좋은 이름으로 변환합니다."""
    return str(name).title()


def is_valid_character_name(name: str) -> bool:
    """Character node 후보로 사용할 수 있는 이름인지 검사합니다.

    제외 기준:
    - 빈 문자열
    - pronoun/quantifier 등 EXCLUDED_CHARACTER_NAMES에 포함된 이름
    - 한 글자 이름
    - 순수 숫자
    """
    if not name or name in EXCLUDED_CHARACTER_NAMES:
        return False
    return len(name) >= 2 and not name.isdigit()


def stable_hash(value: str, length: int = 8) -> str:
    """문자열 기반의 안정적인 짧은 hash를 만듭니다.

    Python 내장 ``hash()``는 실행할 때마다 값이 달라질 수 있으므로 사용하지 않습니다.
    node/edge ID는 재실행해도 같아야 하므로 md5 기반의 짧은 hash를 사용합니다.
    """
    return hashlib.md5(str(value).encode("utf-8")).hexdigest()[:length]


def safe_id_text(value: str) -> str:
    """Neo4j ID에 넣기 좋은 읽기 쉬운 문자열 조각을 만듭니다.

    예:
    - ``king's chambers`` -> ``king_s_chambers``
    - ``small-pool of water`` -> ``small_pool_of_water``
    """
    value = normalize_name(value)
    value = value.replace("'", "_").replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def entity_id(prefix: str, name: str, use_hash: bool = True) -> str:
    """공통 엔티티 ID를 생성합니다.

    node와 edge는 반드시 같은 ID 생성 함수를 사용해야 합니다.
    ``object_id`` 오류의 대부분은 node는 새 ID 규칙, edge는 옛 ID 규칙을 사용해서 생깁니다.

    ``use_hash=True``일 때:
    - loc__kings_bedroom__xxxxxxxx
    - obj__alarm_horn__yyyyyyyy

    hash suffix를 붙이면 apostrophe/hyphen 처리로 인한 충돌을 줄일 수 있습니다.
    """
    norm = normalize_name(name)
    base = safe_id_text(norm)
    if use_hash:
        return f"{prefix}__{base}__{stable_hash(norm)}"
    return f"{prefix}__{base}"


def character_id(name: str) -> str:
    """Character node ID를 생성합니다.

    character는 사용자가 직접 질의할 핵심 엔티티이므로 ``char__king``처럼 읽기 쉬운 ID를 유지합니다.
    """
    return entity_id("char", name, use_hash=False)


def location_id(name: str) -> str:
    """Location node ID를 생성합니다.

    location 이름은 apostrophe/hyphen 변형이 많으므로 hash suffix를 붙입니다.
    """
    return entity_id("loc", name, use_hash=True)


def object_id(name: str) -> str:
    """Object node ID를 생성합니다.

    가장 중요합니다. ``nodes_objects.csv``와 ``edges_episode_contains_object.csv``가 반드시 이 함수를 같이 써야 합니다.
    """
    return entity_id("obj", name, use_hash=True)


def utterance_id(episode_id: str, turn_index: int) -> str:
    """Episode ID와 turn index로 Utterance node ID를 생성합니다."""
    return f"utt__{episode_id}__{turn_index:03d}"


def load_pickle(path: Path) -> Any:
    """pickle 파일을 로드합니다."""
    with path.open("rb") as f:
        return pickle.load(f)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """dict row들을 JSONL 파일로 저장합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =============================================================================
# 3. LIGHT 원본 필드 추출 함수
# =============================================================================


def check_raw_files(raw_dir: Path) -> None:
    """raw 폴더에 필요한 pkl 3개가 있는지 확인합니다.

    실제 전처리에 직접 사용하는 것은 light_data.pkl, light_unseen_data.pkl입니다.
    light_environment.pkl은 현재 스크립트에서는 필수 참조 사전 확인용으로만 존재를 확인합니다.
    """
    required = ["light_data.pkl", "light_unseen_data.pkl", "light_environment.pkl"]
    missing = [name for name in required if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"raw 폴더에 필요한 파일이 없습니다: {missing}. 현재 raw_dir={raw_dir}"
        )


def iter_episodes(raw_dir: Path) -> List[Dict[str, Any]]:
    """train/unseen pkl을 읽고 episode_id, split을 붙여 episode list를 반환합니다.

    episode_id 예:
    - train_ep_000000
    - unseen_ep_000000
    """
    data_files = [
        ("train", raw_dir / "light_data.pkl"),
        ("unseen", raw_dir / "light_unseen_data.pkl"),
    ]

    episodes: List[Dict[str, Any]] = []
    for split, path in data_files:
        data = load_pickle(path)
        for idx, ep in enumerate(data):
            row = dict(ep)
            row["episode_id"] = f"{split}_ep_{idx:06d}"
            row["split"] = split
            episodes.append(row)
    return episodes


def agent_names(ep: Dict[str, Any]) -> List[str]:
    """episode-level 등장 character를 추출합니다.

    원본 필드:
    - ep['agents']

    이 값은 ``APPEARS_IN`` edge와 ``appearance_count`` 계산에 사용합니다.
    """
    names: List[str] = []
    for agent in ep.get("agents") or []:
        if isinstance(agent, dict):
            name = normalize_name(agent.get("name"))
        else:
            name = normalize_name(agent)
        if is_valid_character_name(name):
            names.append(name)
    return names


def speaker_names(ep: Dict[str, Any]) -> List[str]:
    """turn-level speaker 목록을 추출합니다.

    원본 필드:
    - ep['character']

    중복을 제거하지 않습니다. 같은 character가 여러 번 말하면 speaker_turn_count도 여러 번 증가해야 합니다.
    """
    names: List[str] = []
    for raw_name in ep.get("character") or []:
        name = normalize_name(raw_name)
        if is_valid_character_name(name):
            names.append(name)
    return names


def speech_turns(ep: Dict[str, Any]) -> List[str]:
    """episode의 발화문 목록을 문자열 list로 반환합니다."""
    return [str(text) for text in ep.get("speech") or []]


def get_location_info(ep: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    """episode setting에서 location 정보를 추출합니다.

    반환값:
    - normalized_name
    - canonical_name
    - category
    - description
    - background
    """
    setting = ep.get("setting") or {}
    raw_name = setting.get("name") or ""
    norm = normalize_name(raw_name)
    canonical = str(raw_name).strip()
    return (
        norm,
        canonical,
        setting.get("category") or "",
        setting.get("description") or "",
        setting.get("background") or "",
    )


def flatten_room_objects(ep: Dict[str, Any]) -> List[str]:
    """turn-level room_objects를 episode 단위 object list로 펼칩니다.

    room_objects는 보통 turn마다 object list가 들어 있습니다.
    이 함수는 모든 turn의 object를 하나의 list로 모으고 normalize합니다.
    """
    objects: List[str] = []
    for turn_objects in ep.get("room_objects") or []:
        values = turn_objects if isinstance(turn_objects, (list, tuple, set)) else [turn_objects]
        for obj in values:
            name = normalize_name(obj)
            if name:
                objects.append(name)
    return objects


# =============================================================================
# 4. Mention matching
# =============================================================================


def build_mention_index(names: Iterable[str]) -> Dict[str, List[Tuple[str, ...]]]:
    """character 이름 mention 탐지를 위한 index를 만듭니다.

    이름을 첫 token 기준으로 묶어 두면 모든 character를 매번 순회하지 않아도 됩니다.
    예: ``court wizard``는 first token ``court`` 아래에 저장됩니다.
    """
    index: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
    for name in set(names):
        if not is_valid_character_name(name):
            continue
        tokens = tuple(name.split())
        if 0 < len(tokens) <= 5:
            index[tokens[0]].append(tokens)

    # 긴 이름을 먼저 검사해야 ``castle guard``가 ``guard``보다 먼저 잡힙니다.
    for first_token in index:
        index[first_token].sort(key=len, reverse=True)
    return index


def count_mentions(text: str, mention_index: Dict[str, List[Tuple[str, ...]]]) -> Counter:
    """텍스트 안에서 character 이름 mention 횟수를 token 단위로 계산합니다."""
    counts: Counter = Counter()
    tokens = normalize_name(text).split()
    for i, token in enumerate(tokens):
        for candidate in mention_index.get(token, []):
            n = len(candidate)
            if tuple(tokens[i:i + n]) == candidate:
                counts[" ".join(candidate)] += 1
    return counts


def build_character_candidates(episodes: List[Dict[str, Any]]) -> Tuple[set, Dict[str, List[Tuple[str, ...]]]]:
    """agents/speakers에서 관찰된 character 후보와 mention index를 생성합니다."""
    candidates = {FOCUS_CHARACTER}
    for ep in episodes:
        candidates.update(agent_names(ep))
        candidates.update(speaker_names(ep))
    candidates = {name for name in candidates if is_valid_character_name(name)}
    return candidates, build_mention_index(candidates)


# =============================================================================
# 5. King episode 분류 및 GraphRAG 문서 생성
# =============================================================================


def classify_king_episode(ep: Dict[str, Any], mention_index) -> Dict[str, Any]:
    """episode 하나의 king 관련 flag를 계산합니다.

    strong 기준:
    - agents에 king이 직접 등장하거나
    - king이 speaker이거나
    - mention 기반 relevance score가 2 이상이면 strong으로 봅니다.
    """
    agents = set(agent_names(ep))
    speakers = speaker_names(ep)
    mentions = count_mentions("\n".join(speech_turns(ep)), mention_index)

    direct = FOCUS_CHARACTER in agents
    speaker = FOCUS_CHARACTER in speakers
    mention_count = mentions.get(FOCUS_CHARACTER, 0)
    mention = mention_count > 0
    score = int(direct) * 5 + int(speaker) * 3 + min(mention_count, 5)

    if direct:
        relation_type = "direct_appearance"
    elif speaker:
        relation_type = "speaker"
    elif mention:
        relation_type = "mention_only"
    else:
        relation_type = "none"

    return {
        "king_direct_appearance": direct,
        "king_speaker": speaker,
        "king_mention": mention,
        "king_mention_count": mention_count,
        "king_relevance_score": score,
        "is_king_related": direct or speaker or mention,
        "is_king_strong": direct or speaker or score >= 2,
        "king_relation_type": relation_type,
    }


def build_dialogue_text(ep: Dict[str, Any]) -> str:
    """episode dialogue를 사람이 읽기 좋은 text로 변환합니다."""
    speakers = speaker_names(ep)
    speeches = speech_turns(ep)
    emotes = list(ep.get("emote") or [])
    actions = list(ep.get("action") or [])

    lines: List[str] = []
    for i, speech in enumerate(speeches):
        speaker = speakers[i] if i < len(speakers) else "unknown"
        extras = []
        if i < len(emotes) and emotes[i]:
            extras.append(f"emote={emotes[i]}")
        if i < len(actions) and actions[i]:
            extras.append(f"action={actions[i]}")
        suffix = f" ({'; '.join(extras)})" if extras else ""
        lines.append(f"[Turn {i}] {display_name(speaker)}: {speech}{suffix}")
    return "\n".join(lines)


def build_graphrag_doc(ep: Dict[str, Any], flags: Dict[str, Any]) -> Dict[str, Any]:
    """king strong episode 하나를 GraphRAG 입력용 JSON document로 변환합니다."""
    loc_norm, loc_name, loc_cat, loc_desc, loc_bg = get_location_info(ep)
    agents = sorted(set(agent_names(ep)))
    objects = [name for name, _ in Counter(flatten_room_objects(ep)).most_common(20)]

    parts = [
        f"Episode ID: {ep['episode_id']}",
        f"Split: {ep['split']}",
        "",
        "[Setting]",
        f"Name: {loc_name}",
        f"Category: {loc_cat}",
    ]
    if loc_desc:
        parts.append(f"Description: {loc_desc}")
    if loc_bg:
        parts.append(f"Background: {loc_bg}")

    # LLM/RAG가 episode context를 한 번에 이해할 수 있도록 구조화된 섹션을 넣습니다.
    parts += [
        "",
        "[Characters]",
        ", ".join(display_name(name) for name in agents),
        "",
        "[Objects]",
        ", ".join(display_name(name) for name in objects),
        "",
        "[King Relevance]",
        f"Relation type: {flags['king_relation_type']}",
        f"Direct appearance: {flags['king_direct_appearance']}",
        f"King speaker: {flags['king_speaker']}",
        f"King mention count: {flags['king_mention_count']}",
        f"King relevance score: {flags['king_relevance_score']}",
        "",
        "[Dialogue]",
        build_dialogue_text(ep),
    ]

    return {
        "doc_id": f"doc__{ep['episode_id']}",
        "episode_id": ep["episode_id"],
        "title": f"King episode in {loc_name or 'Unknown Location'}",
        "text": "\n".join(parts),
        "metadata": {
            "split": ep["split"],
            "setting_name": loc_name,
            "setting_normalized_name": loc_norm,
            "setting_category": loc_cat,
            "agents": agents,
            "top_objects": objects,
            "utterance_count": len(speech_turns(ep)),
            **flags,
        },
    }


def is_story_relevant_object(name: str) -> bool:
    """object가 왕/궁정/모험 서사에서 의미 있는 후보인지 flag를 반환합니다."""
    tokens = set(name.split())
    return name in STORY_RELEVANT_OBJECT_KEYWORDS or bool(tokens & STORY_RELEVANT_OBJECT_KEYWORDS)


# =============================================================================
# 6. 전체 산출물 생성
# =============================================================================


def build_outputs(raw_dir: Path, out_dir: Path) -> Dict[str, int]:
    """전체 전처리 파이프라인을 실행하고 모든 결과 파일을 저장합니다.

    생성되는 주요 폴더:
    - metrics: 분석 지표 CSV
    - subsets: king 관련 episode JSONL
    - neo4j: node/edge CSV
    - graphrag: RAG 입력용 JSONL
    """
    check_raw_files(raw_dir)

    metrics_dir = out_dir / "metrics"
    subsets_dir = out_dir / "subsets"
    neo4j_dir = out_dir / "neo4j"
    graphrag_dir = out_dir / "graphrag"
    for d in [metrics_dir, subsets_dir, neo4j_dir, graphrag_dir]:
        d.mkdir(parents=True, exist_ok=True)

    episodes = iter_episodes(raw_dir)
    character_names, mention_index = build_character_candidates(episodes)

    # -------------------------------------------------------------------------
    # 6-1. Character ranking + King subset 생성
    # -------------------------------------------------------------------------
    appearance = Counter()
    speaker_turns = Counter()
    mention_counts = Counter()
    unique_episode = Counter()

    episode_flags: Dict[str, Dict[str, Any]] = {}
    subset_all, subset_strong, subset_weak = [], [], []

    for ep in episodes:
        agents = set(agent_names(ep))
        speakers = speaker_names(ep)
        mentions = count_mentions("\n".join(speech_turns(ep)), mention_index)

        appearance.update(agents)
        speaker_turns.update(speakers)
        mention_counts.update(mentions)

        # 같은 episode에서 여러 번 언급되어도 unique_episode는 1만 증가합니다.
        for name in agents | set(speakers) | set(mentions):
            unique_episode[name] += 1

        flags = classify_king_episode(ep, mention_index)
        episode_flags[ep["episode_id"]] = flags

        if flags["is_king_related"]:
            record = {
                "episode_id": ep["episode_id"],
                "split": ep["split"],
                **flags,
                "agents": sorted(agents),
                "setting": ep.get("setting"),
                "character": list(ep.get("character") or []),
                "speech": list(ep.get("speech") or []),
                "emote": list(ep.get("emote") or []),
                "action": list(ep.get("action") or []),
                "room_objects": list(ep.get("room_objects") or []),
                "room_agents": list(ep.get("room_agents") or []),
            }
            subset_all.append(record)
            if flags["is_king_strong"]:
                subset_strong.append(record)
            else:
                subset_weak.append(record)

    # Character node와 metrics가 같은 원천을 쓰도록 한 번에 생성합니다.
    all_names = set(appearance) | set(speaker_turns) | set(mention_counts) | character_names
    rows = []
    for name in sorted(all_names):
        total = appearance[name] * 3 + speaker_turns[name] + mention_counts[name]
        rows.append({
            "character_id": character_id(name),
            "normalized_name": name,
            "canonical_name": display_name(name),
            "appearance_count": appearance[name],
            "speaker_turn_count": speaker_turns[name],
            "mention_count": mention_counts[name],
            "unique_episode_count": unique_episode[name],
            "total_score": total,
            "is_focus_character": name == FOCUS_CHARACTER,
            "is_generic_name": name in GENERIC_NAMES,
            "source": "light_combined",
        })

    chars_df = pd.DataFrame(rows).sort_values(
        ["total_score", "appearance_count", "speaker_turn_count"], ascending=False
    ).reset_index(drop=True)
    chars_df["overall_rank"] = chars_df.index + 1

    chars_df.to_csv(metrics_dir / "top_characters_overall.csv", index=False)
    chars_df[chars_df["normalized_name"] != FOCUS_CHARACTER].to_csv(
        metrics_dir / "top_non_king_characters.csv", index=False
    )

    king_metrics = [
        {
            "episode_id": ep["episode_id"],
            "split": ep["split"],
            **episode_flags[ep["episode_id"]],
            "utterance_count": len(speech_turns(ep)),
        }
        for ep in episodes
    ]
    pd.DataFrame(king_metrics).to_csv(metrics_dir / "king_episode_metrics.csv", index=False)

    write_jsonl(subsets_dir / "king_related_all_episodes.jsonl", subset_all)
    write_jsonl(subsets_dir / "king_related_strong_episodes.jsonl", subset_strong)
    write_jsonl(subsets_dir / "king_related_weak_mention_episodes.jsonl", subset_weak)

    # -------------------------------------------------------------------------
    # 6-2. Neo4j node/edge CSV 생성
    # -------------------------------------------------------------------------
    nodes_characters = chars_df[[
        "character_id", "normalized_name", "canonical_name", "is_focus_character",
        "is_generic_name", "appearance_count", "speaker_turn_count", "mention_count",
        "unique_episode_count", "total_score", "overall_rank", "source",
    ]].copy()

    node_episodes, node_utterances = [], []
    location_meta: Dict[str, Dict[str, Any]] = {}
    object_counter, object_episode_counter = Counter(), Counter()

    edge_appears, edge_speaks, edge_mentions = [], [], []
    edge_occurs, edge_objects = [], []
    co_counter = Counter()

    for ep in episodes:
        eid = ep["episode_id"]
        flags = episode_flags[eid]
        agents = set(agent_names(ep))
        speakers = speaker_names(ep)
        speeches = speech_turns(ep)
        loc_norm, loc_name, loc_cat, loc_desc, loc_bg = get_location_info(ep)

        node_episodes.append({
            "episode_id": eid,
            "split": ep["split"],
            "location_id": location_id(loc_norm) if loc_norm else "",
            "setting_name": loc_name,
            "setting_category": loc_cat,
            "utterance_count": len(speeches),
            "agent_count": len(agents),
            **flags,
        })

        if loc_norm:
            # location node와 OCCURS_IN edge가 같은 location_id()를 사용해야 합니다.
            location_meta.setdefault(loc_norm, {
                "location_id": location_id(loc_norm),
                "normalized_name": loc_norm,
                "canonical_name": loc_name,
                "category": loc_cat,
                "description": loc_desc,
                "background": loc_bg,
            })
            edge_occurs.append({
                "episode_id": eid,
                "location_id": location_id(loc_norm),
                "location_name": loc_norm,
                "location_category": loc_cat,
            })

        for name in sorted(agents):
            edge_appears.append({
                "character_id": character_id(name),
                "episode_id": eid,
                "character_name": name,
                "source_field": "agents",
            })

        # 같은 episode의 agents 조합으로 co-occurrence edge를 누적합니다.
        for left, right in combinations(sorted(agents), 2):
            co_counter[(left, right)] += 1

        emotes = list(ep.get("emote") or [])
        actions = list(ep.get("action") or [])
        for i, text in enumerate(speeches):
            speaker = speakers[i] if i < len(speakers) else ""
            uid = utterance_id(eid, i)

            node_utterances.append({
                "utterance_id": uid,
                "episode_id": eid,
                "turn_index": i,
                "speaker_normalized_name": speaker,
                "speaker_character_id": character_id(speaker) if speaker else "",
                "text": text,
                "emote": emotes[i] if i < len(emotes) else "",
                "action": actions[i] if i < len(actions) else "",
                "is_king_speaker": speaker == FOCUS_CHARACTER,
                "contains_king_mention": count_mentions(text, mention_index).get(FOCUS_CHARACTER, 0) > 0,
            })

            if speaker:
                edge_speaks.append({
                    "character_id": character_id(speaker),
                    "utterance_id": uid,
                    "episode_id": eid,
                    "turn_index": i,
                })

            for mentioned, count in count_mentions(text, mention_index).items():
                edge_mentions.append({
                    "utterance_id": uid,
                    "character_id": character_id(mentioned),
                    "mentioned_name": mentioned,
                    "mention_count": count,
                    "episode_id": eid,
                    "turn_index": i,
                    "is_focus_character_mention": mentioned == FOCUS_CHARACTER,
                })

        objects = flatten_room_objects(ep)
        object_counter.update(objects)
        for obj in set(objects):
            object_episode_counter[obj] += 1

        # 오류 방지 핵심: object edge도 object_id(obj)를 사용합니다.
        # node 생성에서도 아래와 같은 object_id(obj)를 사용합니다.
        for obj, count in Counter(objects).items():
            edge_objects.append({
                "episode_id": eid,
                "object_id": object_id(obj),
                "object_name": obj,
                "object_count_in_episode": count,
            })

    nodes_objects = []
    for obj, count in object_counter.items():
        nodes_objects.append({
            "object_id": object_id(obj),
            "normalized_name": obj,
            "canonical_name": display_name(obj),
            "object_count": count,
            "episode_count": object_episode_counter[obj],
            "is_background_object": obj in BACKGROUND_OBJECTS,
            "is_story_relevant_candidate": is_story_relevant_object(obj),
        })

    co_edges = []
    for (left, right), count in co_counter.items():
        co_edges.append({
            "source_character_id": character_id(left),
            "target_character_id": character_id(right),
            "source_character_name": left,
            "target_character_name": right,
            "co_episode_count": count,
            "is_focus_character_pair": FOCUS_CHARACTER in {left, right},
        })

    outputs = {
        "nodes_characters.csv": nodes_characters,
        "nodes_episodes.csv": pd.DataFrame(node_episodes),
        "nodes_utterances.csv": pd.DataFrame(node_utterances),
        "nodes_locations.csv": pd.DataFrame(list(location_meta.values())),
        "nodes_objects.csv": pd.DataFrame(nodes_objects),
        "edges_character_appears_in_episode.csv": pd.DataFrame(edge_appears).drop_duplicates(["character_id", "episode_id"]),
        "edges_character_speaks_utterance.csv": pd.DataFrame(edge_speaks).drop_duplicates(["character_id", "utterance_id"]),
        "edges_utterance_mentions_character.csv": pd.DataFrame(edge_mentions).drop_duplicates(["utterance_id", "character_id"]),
        "edges_episode_occurs_in_location.csv": pd.DataFrame(edge_occurs).drop_duplicates(["episode_id", "location_id"]),
        "edges_episode_contains_object.csv": pd.DataFrame(edge_objects).drop_duplicates(["episode_id", "object_id"]),
        "edges_character_co_occurs_with_character.csv": pd.DataFrame(co_edges).drop_duplicates(["source_character_id", "target_character_id"]),
    }

    for filename, df in outputs.items():
        df.to_csv(neo4j_dir / filename, index=False)

    # -------------------------------------------------------------------------
    # 6-3. 추가 분석 metrics 생성
    # -------------------------------------------------------------------------
    pd.DataFrame(co_edges).sort_values("co_episode_count", ascending=False).to_csv(
        metrics_dir / "king_cocharacters.csv", index=False
    )

    loc_counts = Counter()
    obj_counts = Counter()
    for ep in episodes:
        if not episode_flags[ep["episode_id"]]["is_king_strong"]:
            continue
        loc_norm, loc_name, loc_cat, _, _ = get_location_info(ep)
        if loc_norm:
            loc_counts[(loc_norm, loc_name, loc_cat)] += 1
        obj_counts.update(flatten_room_objects(ep))

    pd.DataFrame([
        {
            "normalized_name": n,
            "canonical_name": c,
            "location_category": cat,
            "king_episode_count": cnt,
        }
        for (n, c, cat), cnt in loc_counts.items()
    ]).sort_values("king_episode_count", ascending=False).to_csv(
        metrics_dir / "king_locations.csv", index=False
    )

    pd.DataFrame([
        {
            "normalized_name": o,
            "canonical_name": display_name(o),
            "object_count_in_king_episodes": cnt,
            "is_background_object": o in BACKGROUND_OBJECTS,
            "is_story_relevant_candidate": is_story_relevant_object(o),
        }
        for o, cnt in obj_counts.items()
    ]).sort_values("object_count_in_king_episodes", ascending=False).to_csv(
        metrics_dir / "king_objects.csv", index=False
    )

    # -------------------------------------------------------------------------
    # 6-4. GraphRAG JSONL 생성
    # -------------------------------------------------------------------------
    docs = [
        build_graphrag_doc(ep, episode_flags[ep["episode_id"]])
        for ep in episodes
        if episode_flags[ep["episode_id"]]["is_king_strong"]
    ]
    write_jsonl(graphrag_dir / "graphrag_documents_king_strong.jsonl", docs)

    # -------------------------------------------------------------------------
    # 6-5. 최종 검증 및 요약 저장
    # -------------------------------------------------------------------------
    summary = validate_outputs(out_dir)
    with (out_dir / "preprocessing_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # zip 파일은 OUTPUT 내부가 아니라 프로젝트 루트에 둡니다.
    # OUTPUT 안에 zip을 넣고 OUTPUT 전체를 압축하면 자기 자신을 포함할 위험이 있습니다.
    package_path = out_dir.parent / "light_graphrag_preprocessing_outputs.zip"
    if package_path.exists():
        package_path.unlink()
    with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in out_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(out_dir.parent))

    return summary


# =============================================================================
# 7. 최종 검증 함수
# =============================================================================


def count_jsonl(path: Path) -> int:
    """JSONL 파일의 row 수를 세고 JSON 파싱 가능 여부도 함께 검증합니다."""
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL: {path}, line={line_no}, error={e}") from e
            count += 1
    return count


def assert_unique(df: pd.DataFrame, column: str, label: str) -> None:
    """특정 node ID 컬럼이 unique한지 확인합니다."""
    if not df[column].is_unique:
        examples = df[df[column].duplicated()][column].head(10).tolist()
        raise AssertionError(f"{label}.{column} duplicated. Examples: {examples}")


def assert_subset(values, valid_values, message: str) -> None:
    """edge가 참조하는 ID가 node ID 집합 안에 모두 존재하는지 확인합니다."""
    missing = set(values) - set(valid_values)
    if missing:
        raise AssertionError(f"{message}. Missing examples: {sorted(missing)[:10]}")


def validate_outputs(out_dir: Path) -> Dict[str, int]:
    """전체 산출물의 무결성을 검증하고 요약 통계를 반환합니다.

    검증 항목:
    - node ID 중복 여부
    - edge foreign key 참조 무결성
    - king all = strong + weak
    - king strong episode 수 = GraphRAG document 수
    """
    metrics_dir = out_dir / "metrics"
    subsets_dir = out_dir / "subsets"
    neo4j_dir = out_dir / "neo4j"
    graphrag_dir = out_dir / "graphrag"

    nodes_characters = pd.read_csv(neo4j_dir / "nodes_characters.csv")
    nodes_episodes = pd.read_csv(neo4j_dir / "nodes_episodes.csv")
    nodes_utterances = pd.read_csv(neo4j_dir / "nodes_utterances.csv")
    nodes_locations = pd.read_csv(neo4j_dir / "nodes_locations.csv")
    nodes_objects = pd.read_csv(neo4j_dir / "nodes_objects.csv")

    edges_appears = pd.read_csv(neo4j_dir / "edges_character_appears_in_episode.csv")
    edges_speaks = pd.read_csv(neo4j_dir / "edges_character_speaks_utterance.csv")
    edges_mentions = pd.read_csv(neo4j_dir / "edges_utterance_mentions_character.csv")
    edges_occurs = pd.read_csv(neo4j_dir / "edges_episode_occurs_in_location.csv")
    edges_objects = pd.read_csv(neo4j_dir / "edges_episode_contains_object.csv")
    edges_co = pd.read_csv(neo4j_dir / "edges_character_co_occurs_with_character.csv")
    king_metrics = pd.read_csv(metrics_dir / "king_episode_metrics.csv")

    # 1. Node ID 중복 검증
    assert_unique(nodes_characters, "character_id", "nodes_characters")
    assert_unique(nodes_episodes, "episode_id", "nodes_episodes")
    assert_unique(nodes_utterances, "utterance_id", "nodes_utterances")
    assert_unique(nodes_locations, "location_id", "nodes_locations")
    assert_unique(nodes_objects, "object_id", "nodes_objects")

    # 2. Edge가 참조하는 node ID가 실제 node CSV에 존재하는지 검증
    character_ids = set(nodes_characters["character_id"])
    episode_ids = set(nodes_episodes["episode_id"])
    utterance_ids = set(nodes_utterances["utterance_id"])
    location_ids = set(nodes_locations["location_id"])
    object_ids = set(nodes_objects["object_id"])

    assert_subset(edges_appears["character_id"], character_ids, "APPEARS_IN missing Character")
    assert_subset(edges_appears["episode_id"], episode_ids, "APPEARS_IN missing Episode")
    assert_subset(edges_speaks["character_id"], character_ids, "SPEAKS missing Character")
    assert_subset(edges_speaks["utterance_id"], utterance_ids, "SPEAKS missing Utterance")
    assert_subset(edges_mentions["utterance_id"], utterance_ids, "MENTIONS missing Utterance")
    assert_subset(edges_mentions["character_id"], character_ids, "MENTIONS missing Character")
    assert_subset(edges_occurs["episode_id"], episode_ids, "OCCURS_IN missing Episode")
    assert_subset(edges_occurs["location_id"], location_ids, "OCCURS_IN missing Location")
    assert_subset(edges_objects["episode_id"], episode_ids, "CONTAINS_OBJECT missing Episode")
    assert_subset(edges_objects["object_id"], object_ids, "CONTAINS_OBJECT missing Object")
    assert_subset(edges_co["source_character_id"], character_ids, "CO_OCCURS source missing Character")
    assert_subset(edges_co["target_character_id"], character_ids, "CO_OCCURS target missing Character")

    # 3. King subset과 GraphRAG document 수 일치 검증
    all_count = count_jsonl(subsets_dir / "king_related_all_episodes.jsonl")
    strong_count = count_jsonl(subsets_dir / "king_related_strong_episodes.jsonl")
    weak_count = count_jsonl(subsets_dir / "king_related_weak_mention_episodes.jsonl")
    docs_count = count_jsonl(graphrag_dir / "graphrag_documents_king_strong.jsonl")
    metrics_strong = int(king_metrics["is_king_strong"].sum())

    if all_count != strong_count + weak_count:
        raise AssertionError("king all count must equal strong + weak")
    if strong_count != metrics_strong:
        raise AssertionError("king strong JSONL count must match metrics")
    if docs_count != strong_count:
        raise AssertionError("GraphRAG docs count must match strong episodes")

    return {
        "characters": len(nodes_characters),
        "episodes": len(nodes_episodes),
        "utterances": len(nodes_utterances),
        "locations": len(nodes_locations),
        "objects": len(nodes_objects),
        "edges_appears_in": len(edges_appears),
        "edges_speaks": len(edges_speaks),
        "edges_mentions": len(edges_mentions),
        "edges_occurs_in": len(edges_occurs),
        "edges_contains_object": len(edges_objects),
        "edges_co_occurs_with": len(edges_co),
        "king_related_all": all_count,
        "king_related_strong": strong_count,
        "king_related_weak": weak_count,
        "graphrag_documents_king_strong": docs_count,
    }


# =============================================================================
# 8. CLI entrypoint
# =============================================================================


def main() -> None:
    """CLI 인자를 받아 전체 전처리를 실행합니다."""
    parser = argparse.ArgumentParser(
        description="LIGHT 데이터셋 King 중심 GraphRAG 전처리 스크립트"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="pkl 파일 3개가 들어 있는 raw 폴더. 기본값: 스크립트 위치/raw",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="전처리 결과를 저장할 OUTPUT 폴더. 기본값: 스크립트 위치/OUTPUT",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="실행 전 OUTPUT 폴더를 삭제하고 처음부터 다시 생성합니다.",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    out_dir = args.out_dir.resolve()

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    print(f"RAW_DIR: {raw_dir}")
    print(f"OUT_DIR: {out_dir}")

    summary = build_outputs(raw_dir, out_dir)

    print("Preprocessing completed successfully.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
