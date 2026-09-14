"""
LIGHT 데이터셋 기반 King Top Persona episode 전처리 파이프라인.

목표
----
전체 LIGHT episode 중 agents.name == "king"인 agent를 찾고,
그중 episode_count가 가장 많은 persona를 자동 선택합니다.
현재 데이터 기준으로는 다음 persona가 top persona입니다.

    "I am a king of the whole empire. I give rules and pursuit them. I am brave and fearless."

이 persona를 가진 king agent가 포함된 episode만 추출하여 GraphRAG 후속 구축에 사용할
정형/반정형 산출물을 생성합니다.

프로젝트 폴더 구조
-----------------
project_root/
├─ data/
│  ├─ light_data.pkl
│  ├─ light_unseen_data.pkl
│  └─ light_environment.pkl
└─ data_prep/
   ├─ light_king_top_persona_prep.py
   ├─ light_king_top_persona_prep.ipynb
   └─ OUTPUT/
      ├─ metrics/
      ├─ subsets/
      ├─ neo4j/
      ├─ graphrag/
      └─ preprocessing_summary.json

실행 예시
---------
기본 실행:
    python light_king_top_persona_prep.py

기존 OUTPUT 폴더를 삭제하고 재생성:
    python light_king_top_persona_prep.py --clean

다른 persona를 직접 지정:
    python light_king_top_persona_prep.py --target-persona "..."

설계 원칙
---------
1. 데이터 범위는 top king persona episode로 한정합니다.
2. node와 edge는 같은 실행 안에서 같은 ID 생성 함수를 사용해 생성합니다.
3. object_id/location_id에는 hash suffix를 붙여 ID 충돌을 방지합니다.
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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


# =============================================================================
# 0. 프로젝트 기본 경로
# =============================================================================

# 이 파일은 project_root/data_prep/ 아래에 위치한다고 가정합니다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data"
DEFAULT_OUT_DIR = SCRIPT_DIR / "OUTPUT"


# =============================================================================
# 1. 전역 설정값
# =============================================================================

TARGET_AGENT_NAME = "king"

# 일반명사 성격이 강하지만 LIGHT에서는 실제 character로 쓰일 수 있으므로 제거하지 않고 flag만 남깁니다.
GENERIC_NAMES = {
    "person", "people", "man", "woman", "child", "boy", "girl", "guard",
    "servant", "peasant", "villager", "animal", "horse", "bird", "rat",
    "fish", "family", "wife", "husband", "lord", "lady", "guest", "visitor",
    "father", "mother", "friend", "stranger", "traveler", "farmer",
}

# pronoun, quantifier처럼 character node로 만들면 noise가 되는 이름입니다.
EXCLUDED_CHARACTER_NAMES = {
    "i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his",
    "she", "her", "hers", "we", "us", "our", "they", "them", "their",
    "it", "its", "one", "some", "many", "other", "another", "all", "few",
    "none", "someone", "something", "anyone", "anything", "everyone",
}

BACKGROUND_OBJECTS = {
    "table", "wall", "chair", "window", "door", "floor", "room", "candle",
    "bed", "bench", "shelf", "shelves", "stool", "rug", "carpet", "curtain",
    "curtains", "ceiling", "fireplace", "torch", "lamp", "lantern",
}

STORY_RELEVANT_OBJECT_KEYWORDS = {
    "sword", "crown", "gold", "coin", "coins", "throne", "letter", "scroll",
    "book", "key", "ring", "jewel", "jewels", "gem", "gems", "armor",
    "shield", "dagger", "knife", "potion", "chest", "treasure", "map",
}


# =============================================================================
# 2. 기본 유틸 함수
# =============================================================================


def normalize_name(name: Any) -> str:
    """이름을 비교/집계하기 쉬운 형태로 정규화합니다.

    처리 내용:
    - None은 빈 문자열로 처리
    - 소문자화
    - 앞쪽 관사 the/a/an 반복 제거
    - 특수문자 정리

    예:
    - "The King" -> "king"
    - "an a bar" -> "bar"
    """
    if name is None:
        return ""

    value = str(name).strip().lower()

    # 관사가 중첩된 경우를 고려해 더 이상 변하지 않을 때까지 반복 제거합니다.
    prev = None
    while prev != value:
        prev = value
        value = re.sub(r"^(the|a|an)\s+", "", value)

    value = re.sub(r"[^a-z0-9\s'\-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_persona(persona: Any) -> str:
    """persona 텍스트를 persona별 집계 기준으로 정리합니다.

    persona는 원문 의미가 중요하므로 소문자화하지 않고 공백만 정리합니다.
    같은 persona 문장이 공백 차이 때문에 다른 persona로 세어지는 것을 방지합니다.
    """
    if persona is None:
        return ""
    return re.sub(r"\s+", " ", str(persona).strip())


def display_name(name: str) -> str:
    """정규화된 이름을 CSV/보고서에서 보기 좋은 Title Case로 변환합니다."""
    return str(name).title()


def is_valid_character_name(name: str) -> bool:
    """Character node 후보로 사용할 수 있는 이름인지 검사합니다."""
    if not name or name in EXCLUDED_CHARACTER_NAMES:
        return False
    return len(name) >= 2 and not name.isdigit()


def stable_hash(value: str, length: int = 8) -> str:
    """재실행해도 동일하게 나오는 짧은 hash를 생성합니다.

    Python 내장 hash()는 실행마다 값이 달라질 수 있으므로 사용하지 않습니다.
    """
    return hashlib.md5(str(value).encode("utf-8")).hexdigest()[:length]


def safe_id_text(value: str) -> str:
    """Neo4j ID에 넣기 좋은 읽기 쉬운 문자열 조각을 만듭니다."""
    value = normalize_name(value)
    value = value.replace("'", "_").replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def entity_id(prefix: str, name: str, use_hash: bool = True) -> str:
    """node/edge 공통 엔티티 ID를 생성합니다.

    node와 edge는 반드시 같은 ID 생성 함수를 사용해야 합니다.
    object_id/location_id에 hash suffix를 붙이는 이유는 apostrophe, hyphen, 특수문자 처리 과정에서
    서로 다른 원문이 같은 ID로 합쳐지는 것을 방지하기 위해서입니다.
    """
    norm = normalize_name(name)
    base = safe_id_text(norm)
    if use_hash:
        return f"{prefix}__{base}__{stable_hash(norm)}"
    return f"{prefix}__{base}"


def character_id(name: str) -> str:
    """Character node ID를 생성합니다. character는 사람이 읽기 쉽게 hash를 붙이지 않습니다."""
    return entity_id("char", name, use_hash=False)


def location_id(name: str) -> str:
    """Location node ID를 생성합니다."""
    return entity_id("loc", name, use_hash=True)


def object_id(name: str) -> str:
    """Object node ID를 생성합니다. node와 edge 모두 반드시 이 함수를 사용합니다."""
    return entity_id("obj", name, use_hash=True)


def persona_id(persona: str) -> str:
    """Persona를 식별하기 위한 ID를 생성합니다."""
    persona_norm = normalize_persona(persona)
    return f"persona__{stable_hash(persona_norm, length=12)}"


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
    """data 폴더에 필요한 pkl 3개가 있는지 확인합니다."""
    required = ["light_data.pkl", "light_unseen_data.pkl", "light_environment.pkl"]
    missing = [name for name in required if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"필수 pkl 파일이 없습니다: {missing}. raw_dir={raw_dir}")


def iter_episodes(raw_dir: Path) -> List[Dict[str, Any]]:
    """train/unseen pkl을 읽고 episode_id, split을 붙여 반환합니다."""
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


def extract_agent_record(agent: Any) -> Tuple[str, str]:
    """agent 객체에서 name과 persona를 추출합니다.

    LIGHT 데이터의 agent는 보통 dict이지만, 방어적으로 문자열 형태도 처리합니다.
    반환값은 (normalized_name, normalized_persona)입니다.
    """
    if isinstance(agent, dict):
        return normalize_name(agent.get("name")), normalize_persona(agent.get("persona"))
    return normalize_name(agent), ""


def agent_records(ep: Dict[str, Any]) -> List[Tuple[str, str]]:
    """episode의 agents를 (name, persona) list로 반환합니다."""
    return [extract_agent_record(agent) for agent in (ep.get("agents") or [])]


def agent_names(ep: Dict[str, Any]) -> List[str]:
    """episode-level 등장 character 이름을 추출합니다."""
    names: List[str] = []
    for name, _ in agent_records(ep):
        if is_valid_character_name(name):
            names.append(name)
    return names


def agent_persona_texts(ep: Dict[str, Any]) -> List[str]:
    """episode의 agents persona 텍스트를 추출합니다."""
    return [persona for _, persona in agent_records(ep) if persona]


def speaker_names_by_turn(ep: Dict[str, Any]) -> List[str]:
    """turn index를 보존한 speaker 목록을 반환합니다.

    speaker가 invalid name이면 빈 문자열로 둡니다.
    turn index를 보존해야 utterance_id와 speaker edge가 어긋나지 않습니다.
    """
    speakers: List[str] = []
    for raw_name in ep.get("character") or []:
        name = normalize_name(raw_name)
        speakers.append(name if is_valid_character_name(name) else "")
    return speakers


def speaker_names(ep: Dict[str, Any]) -> List[str]:
    """집계용 speaker 이름 목록을 반환합니다. 빈 speaker는 제거합니다."""
    return [name for name in speaker_names_by_turn(ep) if name]


def speech_turns(ep: Dict[str, Any]) -> List[str]:
    """episode의 발화문 목록을 문자열 list로 반환합니다."""
    return [str(text) for text in ep.get("speech") or []]


def get_location_info(ep: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    """episode setting에서 location 정보를 추출합니다."""
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
    """turn-level room_objects를 episode 단위 object list로 펼칩니다."""
    objects: List[str] = []
    for turn_objects in ep.get("room_objects") or []:
        values = turn_objects if isinstance(turn_objects, (list, tuple, set)) else [turn_objects]
        for obj in values:
            name = normalize_name(obj)
            if name:
                objects.append(name)
    return objects


# =============================================================================
# 4. Top king persona 선택 및 mention matching
# =============================================================================


def count_agent_personas(
    episodes: List[Dict[str, Any]],
    target_agent_name: str = TARGET_AGENT_NAME,
) -> pd.DataFrame:
    """target_agent_name을 가진 agents를 persona별로 episode_count/agent_count 집계합니다.

    episode_count는 같은 episode 안에 같은 persona가 여러 번 나와도 1번만 셉니다.
    agent_count는 agents list 안의 등장 횟수를 그대로 셉니다.
    """
    target = normalize_name(target_agent_name)
    episode_counter: Counter = Counter()
    agent_counter: Counter = Counter()
    examples: Dict[str, Dict[str, Any]] = {}

    for ep in episodes:
        personas_in_episode: Set[str] = set()
        for name, persona in agent_records(ep):
            if name != target:
                continue

            persona_key = persona or "[NO_PERSONA]"
            agent_counter[persona_key] += 1
            personas_in_episode.add(persona_key)

            if persona_key not in examples:
                loc_norm, loc_name, loc_cat, _, _ = get_location_info(ep)
                examples[persona_key] = {
                    "example_episode_id": ep["episode_id"],
                    "example_split": ep["split"],
                    "example_setting_name": loc_name,
                    "example_setting_category": loc_cat,
                }

        for persona_key in personas_in_episode:
            episode_counter[persona_key] += 1

    rows = []
    for persona, episode_count in episode_counter.items():
        rows.append({
            "persona_id": persona_id(persona),
            "agent_name": target,
            "persona": persona,
            "episode_count": episode_count,
            "agent_count": agent_counter[persona],
            **examples[persona],
        })

    df = pd.DataFrame(rows).sort_values(
        ["episode_count", "agent_count", "persona"], ascending=[False, False, True]
    ).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def select_target_persona(persona_counts_df: pd.DataFrame, target_persona: Optional[str] = None) -> str:
    """사용자 지정 persona가 있으면 그 값을 사용하고, 없으면 episode_count 1위 persona를 선택합니다."""
    if target_persona:
        normalized_target = normalize_persona(target_persona)
        if normalized_target not in set(persona_counts_df["persona"]):
            raise ValueError("지정한 target_persona가 데이터에 없습니다.")
        return normalized_target

    if persona_counts_df.empty:
        raise ValueError("target agent에 해당하는 persona가 없습니다.")
    return str(persona_counts_df.iloc[0]["persona"])


def episode_has_target_agent_persona(
    ep: Dict[str, Any],
    target_agent_name: str,
    target_persona: str,
) -> bool:
    """episode agents 안에 target name/persona 조합이 존재하는지 확인합니다."""
    target_name = normalize_name(target_agent_name)
    target_persona_norm = normalize_persona(target_persona)
    return any(
        name == target_name and persona == target_persona_norm
        for name, persona in agent_records(ep)
    )


def count_target_agent_persona_in_episode(
    ep: Dict[str, Any],
    target_agent_name: str,
    target_persona: str,
) -> int:
    """한 episode 안에서 target name/persona 조합 agent가 몇 번 등장하는지 셉니다."""
    target_name = normalize_name(target_agent_name)
    target_persona_norm = normalize_persona(target_persona)
    return sum(
        1 for name, persona in agent_records(ep)
        if name == target_name and persona == target_persona_norm
    )


def build_mention_index(names: Iterable[str]) -> Dict[str, List[Tuple[str, ...]]]:
    """character 이름 mention 탐지를 위한 token index를 만듭니다."""
    index: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
    for name in set(names):
        if not is_valid_character_name(name):
            continue
        tokens = tuple(name.split())
        if 0 < len(tokens) <= 5:
            index[tokens[0]].append(tokens)

    # 긴 이름을 먼저 검사해야 복합 이름이 짧은 이름보다 우선 탐지됩니다.
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


def build_character_candidates(episodes: List[Dict[str, Any]]) -> Tuple[Set[str], Dict[str, List[Tuple[str, ...]]]]:
    """전체 episodes의 agents/speakers에서 character 후보와 mention index를 만듭니다."""
    candidates = {TARGET_AGENT_NAME}
    for ep in episodes:
        candidates.update(agent_names(ep))
        candidates.update(speaker_names(ep))
    candidates = {name for name in candidates if is_valid_character_name(name)}
    return candidates, build_mention_index(candidates)


# =============================================================================
# 5. GraphRAG 문서 생성
# =============================================================================


def build_dialogue_text(ep: Dict[str, Any]) -> str:
    """episode dialogue를 사람이 읽기 좋은 text로 변환합니다."""
    speakers = speaker_names_by_turn(ep)
    speeches = speech_turns(ep)
    emotes = list(ep.get("emote") or [])
    actions = list(ep.get("action") or [])

    lines: List[str] = []
    for i, speech in enumerate(speeches):
        speaker = speakers[i] if i < len(speakers) and speakers[i] else "unknown"
        extras = []
        if i < len(emotes) and emotes[i]:
            extras.append(f"emote={emotes[i]}")
        if i < len(actions) and actions[i]:
            extras.append(f"action={actions[i]}")
        suffix = f" ({'; '.join(extras)})" if extras else ""
        lines.append(f"[Turn {i}] {display_name(speaker)}: {speech}{suffix}")
    return "\n".join(lines)


def build_graphrag_doc(ep: Dict[str, Any], target_persona: str) -> Dict[str, Any]:
    """target king persona episode 하나를 GraphRAG 입력용 JSON document로 변환합니다."""
    loc_norm, loc_name, loc_cat, loc_desc, loc_bg = get_location_info(ep)
    agents = sorted(set(agent_names(ep)))
    personas = agent_persona_texts(ep)
    objects = [name for name, _ in Counter(flatten_room_objects(ep)).most_common(20)]

    parts = [
        f"Episode ID: {ep['episode_id']}",
        f"Split: {ep['split']}",
        f"Target Agent: {TARGET_AGENT_NAME}",
        f"Target Persona: {target_persona}",
        "",
        "[Setting]",
        f"Name: {loc_name}",
        f"Category: {loc_cat}",
    ]
    if loc_desc:
        parts.append(f"Description: {loc_desc}")
    if loc_bg:
        parts.append(f"Background: {loc_bg}")

    parts += [
        "",
        "[Characters]",
        ", ".join(display_name(name) for name in agents),
    ]

    if personas:
        parts += [
            "",
            "[Personas]",
            "\n".join(f"- {persona}" for persona in personas),
        ]

    parts += [
        "",
        "[Objects]",
        ", ".join(display_name(name) for name in objects),
        "",
        "[Dialogue]",
        build_dialogue_text(ep),
    ]

    return {
        "doc_id": f"doc__{ep['episode_id']}",
        "episode_id": ep["episode_id"],
        "title": f"Top king persona episode in {loc_name or 'Unknown Location'}",
        "text": "\n".join(parts),
        "metadata": {
            "split": ep["split"],
            "setting_name": loc_name,
            "setting_normalized_name": loc_norm,
            "setting_category": loc_cat,
            "agents": agents,
            "agent_personas": personas,
            "top_objects": objects,
            "utterance_count": len(speech_turns(ep)),
            "target_agent_name": TARGET_AGENT_NAME,
            "target_persona_id": persona_id(target_persona),
            "target_persona": target_persona,
        },
    }


def is_story_relevant_object(name: str) -> bool:
    """object가 왕/궁정/모험 서사에서 의미 있는 후보인지 flag를 반환합니다."""
    tokens = set(name.split())
    return name in STORY_RELEVANT_OBJECT_KEYWORDS or bool(tokens & STORY_RELEVANT_OBJECT_KEYWORDS)


# =============================================================================
# 6. 전체 산출물 생성
# =============================================================================


def build_outputs(
    raw_dir: Path = DEFAULT_RAW_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    target_agent_name: str = TARGET_AGENT_NAME,
    target_persona: Optional[str] = None,
    clean: bool = False,
) -> Dict[str, Any]:
    """Top king persona episode만 대상으로 전체 전처리 파이프라인을 실행합니다.

    생성 폴더:
    - metrics: persona 집계, character ranking, location/object 분석
    - subsets: target persona episode JSONL
    - neo4j: node/edge CSV
    - graphrag: RAG 입력용 JSONL
    """
    raw_dir = Path(raw_dir).resolve()
    out_dir = Path(out_dir).resolve()

    check_raw_files(raw_dir)

    if clean and out_dir.exists():
        shutil.rmtree(out_dir)

    metrics_dir = out_dir / "metrics"
    subsets_dir = out_dir / "subsets"
    neo4j_dir = out_dir / "neo4j"
    graphrag_dir = out_dir / "graphrag"
    for d in [metrics_dir, subsets_dir, neo4j_dir, graphrag_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_episodes = iter_episodes(raw_dir)

    # 1) 전체 episode에서 king persona 분포를 먼저 계산합니다.
    persona_counts_df = count_agent_personas(all_episodes, target_agent_name)
    selected_persona = select_target_persona(persona_counts_df, target_persona)
    selected_persona_id = persona_id(selected_persona)
    persona_counts_df.to_csv(metrics_dir / "king_agent_persona_episode_counts.csv", index=False)

    # 2) episode_count가 가장 많은 king persona episode만 최종 전처리 대상으로 선택합니다.
    target_episodes = [
        ep for ep in all_episodes
        if episode_has_target_agent_persona(ep, target_agent_name, selected_persona)
    ]

    # 3) mention index는 전체 데이터의 character 후보로 만들되, 산출물은 target episode만 대상으로 생성합니다.
    _, mention_index = build_character_candidates(all_episodes)

    # -------------------------------------------------------------------------
    # 6-1. Character metrics 생성
    # -------------------------------------------------------------------------
    appearance = Counter()
    speaker_turns = Counter()
    mention_counts = Counter()
    unique_episode = Counter()

    subset_rows = []
    for ep in target_episodes:
        agents = set(agent_names(ep))
        speakers = speaker_names(ep)
        mentions = count_mentions("\n".join(speech_turns(ep)), mention_index)

        appearance.update(agents)
        speaker_turns.update(speakers)
        mention_counts.update(mentions)

        # 같은 episode에서 여러 번 언급되어도 unique_episode는 1번만 증가합니다.
        for name in agents | set(speakers) | set(mentions):
            unique_episode[name] += 1

        subset_rows.append({
            "episode_id": ep["episode_id"],
            "split": ep["split"],
            "target_agent_name": normalize_name(target_agent_name),
            "target_persona_id": selected_persona_id,
            "target_persona": selected_persona,
            "target_persona_agent_count_in_episode": count_target_agent_persona_in_episode(
                ep, target_agent_name, selected_persona
            ),
            "agents": sorted(agents),
            "agent_personas": agent_persona_texts(ep),
            "setting": ep.get("setting"),
            "character": list(ep.get("character") or []),
            "speech": list(ep.get("speech") or []),
            "emote": list(ep.get("emote") or []),
            "action": list(ep.get("action") or []),
            "room_objects": list(ep.get("room_objects") or []),
            "room_agents": list(ep.get("room_agents") or []),
        })

    all_names = set(appearance) | set(speaker_turns) | set(mention_counts) | {normalize_name(target_agent_name)}
    character_rows = []
    for name in sorted(all_names):
        total = appearance[name] * 3 + speaker_turns[name] + mention_counts[name]
        character_rows.append({
            "character_id": character_id(name),
            "normalized_name": name,
            "canonical_name": display_name(name),
            "appearance_count": appearance[name],
            "speaker_turn_count": speaker_turns[name],
            "mention_count": mention_counts[name],
            "unique_episode_count": unique_episode[name],
            "total_score": total,
            "is_target_agent_name": name == normalize_name(target_agent_name),
            "is_generic_name": name in GENERIC_NAMES,
            "source": "top_king_persona_subset",
        })

    chars_df = pd.DataFrame(character_rows).sort_values(
        ["total_score", "appearance_count", "speaker_turn_count"], ascending=False
    ).reset_index(drop=True)
    chars_df["overall_rank_in_target_subset"] = chars_df.index + 1
    chars_df.to_csv(metrics_dir / "top_characters_in_target_persona_episodes.csv", index=False)

    target_episode_metrics_df = pd.DataFrame([
        {
            "episode_id": ep["episode_id"],
            "split": ep["split"],
            "target_agent_name": normalize_name(target_agent_name),
            "target_persona_id": selected_persona_id,
            "target_persona": selected_persona,
            "target_persona_agent_count_in_episode": count_target_agent_persona_in_episode(
                ep, target_agent_name, selected_persona
            ),
            "utterance_count": len(speech_turns(ep)),
            "agent_count": len(set(agent_names(ep))),
        }
        for ep in target_episodes
    ])
    target_episode_metrics_df.to_csv(metrics_dir / "target_persona_episode_metrics.csv", index=False)

    write_jsonl(subsets_dir / "target_king_persona_episodes.jsonl", subset_rows)

    # -------------------------------------------------------------------------
    # 6-2. Neo4j node/edge CSV 생성
    # -------------------------------------------------------------------------
    nodes_characters = chars_df[[
        "character_id", "normalized_name", "canonical_name", "is_target_agent_name",
        "is_generic_name", "appearance_count", "speaker_turn_count", "mention_count",
        "unique_episode_count", "total_score", "overall_rank_in_target_subset", "source",
    ]].copy()

    node_personas = pd.DataFrame([{
        "persona_id": selected_persona_id,
        "agent_name": normalize_name(target_agent_name),
        "persona": selected_persona,
        "episode_count": len(target_episodes),
        "is_selected_target_persona": True,
    }])

    node_episodes, node_utterances = [], []
    location_meta: Dict[str, Dict[str, Any]] = {}
    object_counter, object_episode_counter = Counter(), Counter()

    edge_appears, edge_has_persona, edge_speaks, edge_mentions = [], [], [], []
    edge_occurs, edge_objects = [], []
    co_counter = Counter()

    for ep in target_episodes:
        eid = ep["episode_id"]
        agents = set(agent_names(ep))
        speakers_by_turn = speaker_names_by_turn(ep)
        speeches = speech_turns(ep)
        loc_norm, loc_name, loc_cat, loc_desc, loc_bg = get_location_info(ep)
        target_count = count_target_agent_persona_in_episode(ep, target_agent_name, selected_persona)

        node_episodes.append({
            "episode_id": eid,
            "split": ep["split"],
            "location_id": location_id(loc_norm) if loc_norm else "",
            "setting_name": loc_name,
            "setting_category": loc_cat,
            "utterance_count": len(speeches),
            "agent_count": len(agents),
            "target_agent_name": normalize_name(target_agent_name),
            "target_persona_id": selected_persona_id,
            "target_persona_agent_count_in_episode": target_count,
        })

        if loc_norm:
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

        # 선택된 king persona와 episode를 연결하는 edge입니다.
        edge_has_persona.append({
            "episode_id": eid,
            "character_id": character_id(normalize_name(target_agent_name)),
            "persona_id": selected_persona_id,
            "target_persona_agent_count_in_episode": target_count,
        })

        for left, right in combinations(sorted(agents), 2):
            co_counter[(left, right)] += 1

        emotes = list(ep.get("emote") or [])
        actions = list(ep.get("action") or [])
        for i, text in enumerate(speeches):
            speaker = speakers_by_turn[i] if i < len(speakers_by_turn) else ""
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
                "is_target_agent_speaker": speaker == normalize_name(target_agent_name),
                "contains_target_agent_mention": count_mentions(text, mention_index).get(normalize_name(target_agent_name), 0) > 0,
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
                    "is_target_agent_mention": mentioned == normalize_name(target_agent_name),
                })

        objects = flatten_room_objects(ep)
        object_counter.update(objects)
        for obj in set(objects):
            object_episode_counter[obj] += 1

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
            "is_target_agent_pair": normalize_name(target_agent_name) in {left, right},
        })

    neo4j_outputs = {
        "nodes_characters.csv": nodes_characters,
        "nodes_personas.csv": node_personas,
        "nodes_episodes.csv": pd.DataFrame(node_episodes),
        "nodes_utterances.csv": pd.DataFrame(node_utterances),
        "nodes_locations.csv": pd.DataFrame(list(location_meta.values())),
        "nodes_objects.csv": pd.DataFrame(nodes_objects),
        "edges_character_appears_in_episode.csv": pd.DataFrame(edge_appears).drop_duplicates(["character_id", "episode_id"]),
        "edges_episode_has_target_persona.csv": pd.DataFrame(edge_has_persona).drop_duplicates(["episode_id", "character_id", "persona_id"]),
        "edges_character_speaks_utterance.csv": pd.DataFrame(edge_speaks).drop_duplicates(["character_id", "utterance_id"]),
        "edges_utterance_mentions_character.csv": pd.DataFrame(edge_mentions).drop_duplicates(["utterance_id", "character_id"]),
        "edges_episode_occurs_in_location.csv": pd.DataFrame(edge_occurs).drop_duplicates(["episode_id", "location_id"]),
        "edges_episode_contains_object.csv": pd.DataFrame(edge_objects).drop_duplicates(["episode_id", "object_id"]),
        "edges_character_co_occurs_with_character.csv": pd.DataFrame(co_edges).drop_duplicates(["source_character_id", "target_character_id"]),
    }

    for filename, df in neo4j_outputs.items():
        df.to_csv(neo4j_dir / filename, index=False)

    # -------------------------------------------------------------------------
    # 6-3. 분석 metrics 생성
    # -------------------------------------------------------------------------
    pd.DataFrame(co_edges).sort_values("co_episode_count", ascending=False).to_csv(
        metrics_dir / "target_persona_cocharacters.csv", index=False
    )

    loc_counts = Counter()
    obj_counts = Counter()
    for ep in target_episodes:
        loc_norm, loc_name, loc_cat, _, _ = get_location_info(ep)
        if loc_norm:
            loc_counts[(loc_norm, loc_name, loc_cat)] += 1
        obj_counts.update(flatten_room_objects(ep))

    pd.DataFrame([
        {
            "normalized_name": n,
            "canonical_name": c,
            "location_category": cat,
            "target_persona_episode_count": cnt,
        }
        for (n, c, cat), cnt in loc_counts.items()
    ]).sort_values("target_persona_episode_count", ascending=False).to_csv(
        metrics_dir / "target_persona_locations.csv", index=False
    )

    pd.DataFrame([
        {
            "normalized_name": o,
            "canonical_name": display_name(o),
            "object_count_in_target_persona_episodes": cnt,
            "is_background_object": o in BACKGROUND_OBJECTS,
            "is_story_relevant_candidate": is_story_relevant_object(o),
        }
        for o, cnt in obj_counts.items()
    ]).sort_values("object_count_in_target_persona_episodes", ascending=False).to_csv(
        metrics_dir / "target_persona_objects.csv", index=False
    )

    target_summary = {
        "target_agent_name": normalize_name(target_agent_name),
        "target_persona_id": selected_persona_id,
        "target_persona": selected_persona,
        "target_episode_count": len(target_episodes),
        "raw_total_episode_count": len(all_episodes),
    }
    with (metrics_dir / "target_persona_summary.json").open("w", encoding="utf-8") as f:
        json.dump(target_summary, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------------------
    # 6-4. GraphRAG JSONL 생성
    # -------------------------------------------------------------------------
    docs = [build_graphrag_doc(ep, selected_persona) for ep in target_episodes]
    write_jsonl(graphrag_dir / "graphrag_documents_target_king_persona.jsonl", docs)

    # -------------------------------------------------------------------------
    # 6-5. 최종 검증 및 요약 저장
    # -------------------------------------------------------------------------
    summary = validate_outputs(out_dir)
    summary.update(target_summary)
    with (out_dir / "preprocessing_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # zip 파일은 OUTPUT 내부가 아니라 data_prep 폴더에 둡니다.
    package_path = out_dir.parent / "light_king_top_persona_preprocessing_outputs.zip"
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
    """JSONL 파일 row 수를 세고 각 line이 JSON으로 파싱되는지 검증합니다."""
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
    """특정 ID 컬럼이 unique한지 확인합니다."""
    if not df[column].is_unique:
        examples = df[df[column].duplicated()][column].head(10).tolist()
        raise AssertionError(f"{label}.{column} duplicated. Examples: {examples}")


def assert_subset(values, valid_values, message: str) -> None:
    """edge가 참조하는 ID가 node ID 집합 안에 모두 존재하는지 확인합니다."""
    missing = set(values) - set(valid_values)
    missing = {x for x in missing if pd.notna(x) and str(x) != ""}
    if missing:
        raise AssertionError(f"{message}. Missing examples: {sorted(missing)[:10]}")


def validate_outputs(out_dir: Path) -> Dict[str, int]:
    """전체 산출물의 무결성을 검증하고 요약 통계를 반환합니다."""
    metrics_dir = out_dir / "metrics"
    subsets_dir = out_dir / "subsets"
    neo4j_dir = out_dir / "neo4j"
    graphrag_dir = out_dir / "graphrag"

    nodes_characters = pd.read_csv(neo4j_dir / "nodes_characters.csv")
    nodes_personas = pd.read_csv(neo4j_dir / "nodes_personas.csv")
    nodes_episodes = pd.read_csv(neo4j_dir / "nodes_episodes.csv")
    nodes_utterances = pd.read_csv(neo4j_dir / "nodes_utterances.csv")
    nodes_locations = pd.read_csv(neo4j_dir / "nodes_locations.csv")
    nodes_objects = pd.read_csv(neo4j_dir / "nodes_objects.csv")

    edges_appears = pd.read_csv(neo4j_dir / "edges_character_appears_in_episode.csv")
    edges_has_persona = pd.read_csv(neo4j_dir / "edges_episode_has_target_persona.csv")
    edges_speaks = pd.read_csv(neo4j_dir / "edges_character_speaks_utterance.csv")
    edges_mentions = pd.read_csv(neo4j_dir / "edges_utterance_mentions_character.csv")
    edges_occurs = pd.read_csv(neo4j_dir / "edges_episode_occurs_in_location.csv")
    edges_objects = pd.read_csv(neo4j_dir / "edges_episode_contains_object.csv")
    edges_co = pd.read_csv(neo4j_dir / "edges_character_co_occurs_with_character.csv")
    target_metrics = pd.read_csv(metrics_dir / "target_persona_episode_metrics.csv")

    # 1. Node ID 중복 검증
    assert_unique(nodes_characters, "character_id", "nodes_characters")
    assert_unique(nodes_personas, "persona_id", "nodes_personas")
    assert_unique(nodes_episodes, "episode_id", "nodes_episodes")
    assert_unique(nodes_utterances, "utterance_id", "nodes_utterances")
    assert_unique(nodes_locations, "location_id", "nodes_locations")
    assert_unique(nodes_objects, "object_id", "nodes_objects")

    # 2. Edge 참조 무결성 검증
    character_ids = set(nodes_characters["character_id"])
    persona_ids = set(nodes_personas["persona_id"])
    episode_ids = set(nodes_episodes["episode_id"])
    utterance_ids = set(nodes_utterances["utterance_id"])
    location_ids = set(nodes_locations["location_id"])
    object_ids = set(nodes_objects["object_id"])

    assert_subset(edges_appears["character_id"], character_ids, "APPEARS_IN missing Character")
    assert_subset(edges_appears["episode_id"], episode_ids, "APPEARS_IN missing Episode")
    assert_subset(edges_has_persona["episode_id"], episode_ids, "HAS_TARGET_PERSONA missing Episode")
    assert_subset(edges_has_persona["character_id"], character_ids, "HAS_TARGET_PERSONA missing Character")
    assert_subset(edges_has_persona["persona_id"], persona_ids, "HAS_TARGET_PERSONA missing Persona")
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

    # 3. target episode 수와 JSONL/doc 수 일치 검증
    subset_count = count_jsonl(subsets_dir / "target_king_persona_episodes.jsonl")
    docs_count = count_jsonl(graphrag_dir / "graphrag_documents_target_king_persona.jsonl")
    metrics_count = len(target_metrics)

    if subset_count != metrics_count:
        raise AssertionError("target subset JSONL count must match target metrics")
    if docs_count != subset_count:
        raise AssertionError("GraphRAG docs count must match target subset episodes")

    return {
        "characters": len(nodes_characters),
        "personas": len(nodes_personas),
        "episodes": len(nodes_episodes),
        "utterances": len(nodes_utterances),
        "locations": len(nodes_locations),
        "objects": len(nodes_objects),
        "edges_appears_in": len(edges_appears),
        "edges_has_target_persona": len(edges_has_persona),
        "edges_speaks": len(edges_speaks),
        "edges_mentions": len(edges_mentions),
        "edges_occurs_in": len(edges_occurs),
        "edges_contains_object": len(edges_objects),
        "edges_co_occurs_with": len(edges_co),
        "target_persona_episode_jsonl": subset_count,
        "graphrag_documents_target_persona": docs_count,
    }


# =============================================================================
# 8. CLI entrypoint
# =============================================================================


def main() -> None:
    """CLI 인자를 받아 전체 전처리를 실행합니다."""
    parser = argparse.ArgumentParser(
        description="LIGHT 데이터셋 top king persona episode 전처리 스크립트"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="pkl 파일 3개가 들어 있는 data 폴더. 기본값: 프로젝트루트/data",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="전처리 결과를 저장할 OUTPUT 폴더. 기본값: 프로젝트루트/data_prep/OUTPUT",
    )
    parser.add_argument(
        "--target-agent-name",
        type=str,
        default=TARGET_AGENT_NAME,
        help="persona별 episode_count를 계산할 agents.name. 기본값: king",
    )
    parser.add_argument(
        "--target-persona",
        type=str,
        default=None,
        help="직접 지정할 persona. 생략하면 episode_count가 가장 많은 persona를 자동 선택합니다.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="실행 전 OUTPUT 폴더를 삭제하고 처음부터 다시 생성합니다.",
    )
    args = parser.parse_args()

    print(f"RAW_DIR: {args.raw_dir.resolve()}")
    print(f"OUT_DIR: {args.out_dir.resolve()}")

    summary = build_outputs(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        target_agent_name=args.target_agent_name,
        target_persona=args.target_persona,
        clean=args.clean,
    )

    print("Preprocessing completed successfully.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
