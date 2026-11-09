import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from thread_classifier import (
    CRITERIA_KEYS,
    aggregate_person_scores,
    classify_all_threads,
)
from thread_digest import build_threads, load_chat_messages


CRITERIA = [
    ("memory_archivist", "Memory Archivist"),
    ("conversation_connector", "Conversation Connector"),
    ("story_weaver", "Story Weaver"),
    ("vibe_lifter", "Vibe Lifter"),
    ("prompt_catalyst", "Prompt Catalyst"),
    ("life_update_anchor", "Life Update Anchor"),
]

INTRO_HINTS = [
    "this is",
    "here's my intro",
    "here is my intro",
    "quick intro",
    "about myself",
    "my turn",
    "post-graduation",
    "happy to connect",
    "i am",
    "i'm",
    "my name",
    "myself",
    "i live",
    "i work",
    "i studied",
    "i grew up",
    "born",
    "based in",
    "currently",
]

EXCLUDED_NAMES = {"system", "me"}


def canonical_person_name(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return ""
    cleaned = re.sub(
        r"\s*(?:JGVV|jGVV|India No|Home|US|Ayyagari|One Of Three|one of three)\b",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def compact_person_name(name: str) -> str:
    raw = canonical_person_name(name)
    if not raw:
        return raw
    parts = raw.split(" ")
    if len(parts) <= 2:
        return raw
    return f"{parts[0]} {parts[1][0].upper()}."


def looks_like_person(name: str) -> bool:
    if not name:
        return False
    return canonical_person_name(name).lower() not in EXCLUDED_NAMES


def detect_mentions(text: str, known_people: set[str]) -> set[str]:
    lower_text = (text or "").lower()
    mentions = set()
    for person in known_people:
        if person.lower() in lower_text:
            mentions.add(person)
    return mentions


def load_media_flags(path: Path) -> list[dict]:
    flags = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# ") or line.startswith("---") or line.startswith("Export date:"):
            continue
        if line.startswith("## "):
            continue

        match = re.match(r"^\[(?P<ts>[^\]]+)\]\s*(?P<body>.*)$", line)
        if not match:
            continue

        body = match.group("body")
        lower = body.lower()
        has_media = "<image omitted>" in lower or "<gif omitted>" in lower or "[image" in lower
        flags.append({"has_media": has_media, "raw_body": body})
    return flags



def choose_intro_message(messages: list[dict]) -> dict | None:
    scored = []
    for msg in messages:
        text = (msg.get("text") or "").strip()
        if len(text) < 35:
            continue
        lower = text.lower()
        score = 0
        if any(hint in lower for hint in INTRO_HINTS):
            score += 50
        if re.search(r"\b(i|i'm|i am|my name|myself|my family|i live|i work|i studied|i grew up|born|based in)\b", lower):
            score += 20
        if len(text) >= 80:
            score += 15
        if any(kw in lower for kw in ["school", "college", "work", "retired", "family", "teacher", "moved", "live", "job", "wife", "husband", "daughter", "son"]):
            score += 10
        if any(kw in lower for kw in ["this is", "here is my intro", "here's my intro", "quick intro", "about myself"]):
            score += 25
        if len(text) > 220:
            score += 10
        if score > 0:
            scored.append((score, msg))

    if not scored:
        return messages[0] if messages else None

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    if len(scored) > 1 and scored[1][0] >= scored[0][0] - 10:
        combined = " ".join(msg.get("text", "") for _, msg in scored[:2] if msg.get("text"))
        return {"text": combined, "author": best.get("author"), "timestamp": best.get("timestamp")}
    return best


_MAX_INTRO_MSGS = 80   # cap to stay inside token budget
_MAX_INTRO_MSG_CHARS = 300


def llm_intro_excerpt(
    client: OpenAI,
    model: str,
    person: str,
    messages: list[dict],
    max_lines: int = 4,
) -> list[str]:
    # Send ALL of the person's messages; let the LLM find the intro semantically.
    texts = []
    for m in messages:
        t = (m.get("text") or "").strip()
        if not t or len(t) < 15:
            continue
        if len(t) > _MAX_INTRO_MSG_CHARS:
            t = t[:_MAX_INTRO_MSG_CHARS] + "…"
        texts.append(t)
    if not texts:
        return []

    joined = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts[:_MAX_INTRO_MSGS]))
    prompt = (
        f"Person name: {person}\n\n"
        "Task: Write a 3-line bio for this person based only on facts they stated about themselves.\n\n"
        "The bio must answer (where the chat provides evidence):\n"
        "  Line 1 – Where they live now and what they do / career.\n"
        "  Line 2 – Family situation (spouse, kids, parents) if mentioned.\n"
        "  Line 3 – Their JGVV / school connection (years attended, section, memories they own).\n\n"
        "Rules:\n"
        "1) Use ONLY facts the person stated about THEMSELVES – ignore any messages where they talk about other people.\n"
        "2) Do not include jokes, reactions, opinions about the group, or nostalgic comments about classmates.\n"
        "3) Write in third person (e.g. 'Lives in Hyderabad, works in IT').\n"
        "4) If evidence for a line is missing, omit that line.\n"
        "5) Maximum 3 lines. Keep each line under 120 characters.\n"
        '6) Return strict JSON only: {"intro_lines":["...","...","..."]}.\n\n'
        f"Messages from {person}:\n"
        f"{joined}\n\n"
        "IMPORTANT: If this person never shared any of the above information about themselves, "
        "return {\"intro_lines\": []} — do not invent content or use messages where they talk about other people."
    )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You extract biography-like self-intro lines from chat logs. Return strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    text = (response.output_text or "").strip()
    if not text:
        return []

    # Strip markdown fences that some models wrap around JSON
    text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text, flags=re.IGNORECASE)

    parsed = json.loads(text)
    lines = parsed.get("intro_lines") if isinstance(parsed, dict) else None
    if not isinstance(lines, list):
        return []

    clean = []
    for line in lines:
        if not isinstance(line, str):
            continue
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            clean.append(line)
        if len(clean) >= max_lines:
            break
    return clean


def intro_excerpt(text: str, max_lines: int = 4) -> list[str]:
    if not text:
        return []
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+", text)
    lines = [part.strip() for part in parts if part.strip()]
    return lines[:max_lines]


def build_insights(
    messages: list[dict],
    source_path: Path,
    llm_client: OpenAI | None = None,
    llm_model: str = "",
    llm_classify: bool = False,
    out_path: Path | None = None,
) -> dict:
    media_flags = load_media_flags(source_path)
    for idx, msg in enumerate(messages):
        flag = media_flags[idx] if idx < len(media_flags) else {"has_media": False, "raw_body": ""}
        msg["has_media"] = bool(flag.get("has_media"))

    known_people: set[str] = set()
    for msg in messages:
        author = canonical_person_name(msg.get("author"))
        if looks_like_person(author):
            known_people.add(author)

    per_person_messages: dict[str, list] = defaultdict(list)
    per_person_images: dict[str, list] = defaultdict(list)

    for msg in messages:
        author = canonical_person_name(msg.get("author"))
        text = (msg.get("text") or "").strip()
        mtype = (msg.get("type") or "text").lower()
        if not looks_like_person(author):
            continue
        if not text and mtype == "system":
            continue
        if "this message was deleted" in text.lower():
            continue
        per_person_messages[author].append(msg)
        if msg.get("has_media"):
            per_person_images[author].append(
                {
                    "id": msg.get("id"),
                    "timestamp": msg.get("timestamp_iso"),
                    "caption": text[:220] if text else "Shared image",
                }
            )

    # --- Criterion scoring: LLM-thread path (preferred) or message-count fallback ---
    threads = build_threads(messages)

    if llm_classify and llm_client and llm_model:
        print(f"  classifying {len(threads)} threads with LLM…", flush=True)
        enriched_threads = classify_all_threads(llm_client, llm_model, threads, verbose=True)
        # Canonicalise LLM-returned names before aggregating
        for thread in enriched_threads:
            raw_scores = thread.get("llm_scores") or {}
            thread["llm_scores"] = {
                canonical_person_name(person): scores
                for person, scores in raw_scores.items()
                if canonical_person_name(person)
            }
        per_person_scores = aggregate_person_scores(enriched_threads)
        # Ensure every known person has an entry
        for person in known_people:
            if person not in per_person_scores:
                per_person_scores[person] = {k: 0 for k in CRITERIA_KEYS}
    else:
        # Minimal heuristic fallback: message-count proxy only
        # Reuse existing LLM scores from a previous run rather than message-count proxy
        existing_scores: dict[str, dict[str, int]] = {}
        if out_path and out_path.exists():
            try:
                prev = json.loads(out_path.read_text(encoding="utf-8"))
                for p in prev.get("people", []):
                    cs = p.get("criteria_scores")
                    if isinstance(cs, dict) and any(v > 0 for v in cs.values()):
                        existing_scores[canonical_person_name(p["name"])] = cs
            except Exception:
                pass

        per_person_scores = {
            person: existing_scores.get(person, {k: 0 for k in CRITERIA_KEYS})
            for person in known_people
        }
        enriched_threads = threads

    # Determine group leaders by criterion
    criterion_leaders = []
    criterion_max = {}
    for key, label in CRITERIA:
        max_score = 0
        for person in known_people:
            max_score = max(max_score, per_person_scores[person][key])
        criterion_max[key] = max_score
        leaders = sorted(
            [person for person in known_people if per_person_scores[person][key] == max_score and max_score > 0]
        )
        criterion_leaders.append(
            {
                "criterion": key,
                "label": label,
                "score": max_score,
                "leaders": leaders,
            }
        )

    people_out = []
    for person in sorted(known_people):
        scores = per_person_scores[person]
        sorted_criteria = sorted(CRITERIA, key=lambda item: scores[item[0]], reverse=True)
        top_non_zero = [key for key, _ in sorted_criteria if scores[key] > 0][:3]
        leader_criteria = []
        for key, label in CRITERIA:
            if criterion_max[key] > 0 and scores[key] == criterion_max[key]:
                leader_criteria.append(
                    {
                        "criterion": key,
                        "label": label,
                        "score": scores[key],
                        "is_group_leader": True,
                    }
                )

        intro_msg = choose_intro_message(per_person_messages[person])
        intro_text = (intro_msg.get("text") if intro_msg else "") or ""

        intro_lines = []
        intro_source = "heuristic"
        if llm_client and llm_model:
            try:
                intro_lines = llm_intro_excerpt(llm_client, llm_model, person, per_person_messages[person], max_lines=4)
                if intro_lines:
                    intro_source = "llm"
            except Exception:
                intro_lines = []

        if not intro_lines:
            intro_lines = intro_excerpt(intro_text, max_lines=4)

        summary_lines = []
        for key in top_non_zero:
            label = dict(CRITERIA)[key]
            if criterion_max[key] > 0 and scores[key] == criterion_max[key]:
                summary_lines.append(f"Leads the chat on {label.lower()} (score {scores[key]}).")
            else:
                summary_lines.append(f"Strong on {label.lower()} (score {scores[key]}).")

        people_out.append(
            {
                "name": person,
                "display_name": compact_person_name(person) if person != "You" else "You (me)",
                "message_count": len(per_person_messages[person]),
                "criteria_scores": scores,
                "leader_criteria": leader_criteria,
                "top_criteria": top_non_zero,
                "summary_lines": summary_lines,
                "intro_card": {
                    "intro_excerpt": intro_lines,
                    "intro_source": intro_source,
                    "images": per_person_images[person][:3],
                },
            }
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source_path),
        "criterion_leaders": criterion_leaders,
        "people": sorted(people_out, key=lambda p: p["message_count"], reverse=True),
        "threads": enriched_threads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build enhanced chat insights for memory lanes")
    parser.add_argument("--chat", default="input/oct31/chat.txt", help="Path to chat txt export")
    parser.add_argument("--out", default="input/oct31/chat_insights.json", help="Path to output insights json")
    parser.add_argument(
        "--thread-out",
        default="input/oct31/thread_digest.json",
        help="Path to output thread digest json",
    )
    parser.add_argument("--llm-intro", action="store_true", help="Use OpenAI LLM to extract intro lines")
    parser.add_argument("--llm-classify", action="store_true", help="Use OpenAI LLM to semantically classify threads and score people")
    parser.add_argument("--llm-model", default="gpt-4.1-mini", help="OpenAI model to use")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    chat_path = (base_dir / args.chat).resolve()
    out_path = (base_dir / args.out).resolve()
    thread_out_path = (base_dir / args.thread_out).resolve()

    messages = load_chat_messages(str(chat_path))

    llm_client = None
    llm_model = ""
    if args.llm_intro or args.llm_classify:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if api_key:
            llm_client = OpenAI(api_key=api_key)
            llm_model = args.llm_model
            print(f"LLM enabled  model={llm_model}  classify={args.llm_classify}  intro={args.llm_intro}")
        else:
            print("OPENAI_API_KEY not set – falling back to heuristic scoring")

    insights = build_insights(
        messages,
        chat_path,
        llm_client=llm_client,
        llm_model=llm_model,
        llm_classify=args.llm_classify,
        out_path=out_path,
    )

    thread_out_path.parent.mkdir(parents=True, exist_ok=True)
    thread_out_path.write_text(
        json.dumps(insights["threads"], ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(insights, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"Loaded {len(messages)} messages")
    print(f"Threads: {len(insights['threads'])} -> {thread_out_path}")
    print(f"People insights: {len(insights['people'])} -> {out_path}")


if __name__ == "__main__":
    main()
