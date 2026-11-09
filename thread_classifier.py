"""
Semantic thread classifier.

Sends each thread to an LLM to:
  1. Assign a human-readable topic label and one-line summary.
  2. Score every participant against 6 criteria on a 0-10 scale.

This replaces the old TOPIC_KEYWORDS / score_message_for_criteria heuristics.
"""

import json
import re
from typing import Any

from openai import OpenAI

CRITERIA = [
    ("memory_archivist",
     "Memory Archivist – names specific classmates, teachers, seating positions, class years, family "
     "connections, or school events with precision. ONE short but accurate name-recall or relationship "
     "fact scores HIGH (7-10). Generic 'I remember' without specifics scores LOW (1-3). "
     "Do NOT penalise short messages – precision beats length."),
    ("conversation_connector",
     "Conversation Connector – directly welcomes newcomers, introduces people to each other, "
     "bridges two separate sub-conversations, or explicitly pulls quiet members in."),
    ("story_weaver",
     "Story Weaver – shares a personal narrative or anecdote with context, detail, and arc "
     "(not just a one-liner reaction). Length and personal depth matter here."),
    ("vibe_lifter",
     "Vibe Lifter – injects humour, warmth, or positive energy that visibly lifts the thread tone. "
     "Emojis alone don't count; look for wit, affection, or playful commentary."),
    ("prompt_catalyst",
     "Prompt Catalyst – poses a question or observation that genuinely opens a new line of "
     "conversation; follow-up questions within an existing sub-thread count less than topic-starters."),
    ("life_update_anchor",
     "Life Update Anchor – shares a concrete personal update (current city, job change, family "
     "milestone, retirement). Must be first-person factual, not speculation about others."),
]

CRITERIA_KEYS = [k for k, _ in CRITERIA]

_SYSTEM = (
    "You analyse WhatsApp group-chat threads and return strict JSON only.\n"
    "No prose, no markdown, no explanation outside the JSON object."
)

_USER_TEMPLATE = """\
Classify this chat thread and score each participant.

CRITERIA (score 0-10 each):
{criteria_block}

SCORING RULES:
- Score WHAT was said, not HOW MUCH. A single precise memory fact = 8-10 for memory_archivist.
- A long message with no specific contribution scores LOW on all criteria.
- Scores are independent – someone can be 9 on memory_archivist and 0 on story_weaver.
- If a person does not contribute meaningfully to a criterion, score them 0, not 1.

THREAD ({n_msgs} messages):
{thread_text}

Return JSON exactly:
{{
  "topic": "<short slug, e.g. school_reunion>",
  "topic_summary": "<one plain-English sentence>",
  "scores": {{
    "<person_name>": {{
      "memory_archivist": <0-10>,
      "conversation_connector": <0-10>,
      "story_weaver": <0-10>,
      "vibe_lifter": <0-10>,
      "prompt_catalyst": <0-10>,
      "life_update_anchor": <0-10>
    }}
  }}
}}
Only include people who actually speak in the thread.
"""

# Trim threads that would blow the context window
_MAX_THREAD_MESSAGES = 60
_MAX_MSG_CHARS = 280


def _render_thread(messages: list[dict]) -> str:
    rendered = []
    for m in messages[:_MAX_THREAD_MESSAGES]:
        author = (m.get("author") or "?").strip()
        text = (m.get("text") or "").strip()
        if not text or author == "system":
            continue
        if len(text) > _MAX_MSG_CHARS:
            text = text[:_MAX_MSG_CHARS] + "…"
        ts = (m.get("timestamp_iso") or "")[:16]
        rendered.append(f"[{ts}] {author}: {text}")
    return "\n".join(rendered)


def _criteria_block() -> str:
    return "\n".join(f"  {k}: {desc}" for k, desc in CRITERIA)


def _zero_scores() -> dict[str, int]:
    return {k: 0 for k in CRITERIA_KEYS}


def classify_thread(
    client: OpenAI,
    model: str,
    thread: dict,
) -> dict[str, Any]:
    """Return topic, topic_summary, and per-person criterion scores for one thread."""
    messages = thread.get("messages") or []
    thread_text = _render_thread(messages)
    if not thread_text.strip():
        return {"topic": "system_notice", "topic_summary": "", "scores": {}}

    speakers = sorted({
        (m.get("author") or "").strip()
        for m in messages
        if (m.get("author") or "").strip() not in ("", "system")
    })

    n_msgs = sum(
        1 for m in messages
        if (m.get("author") or "").strip() not in ("", "system") and m.get("text")
    )

    prompt = _USER_TEMPLATE.format(
        criteria_block=_criteria_block(),
        n_msgs=n_msgs,
        thread_text=thread_text,
    )

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,
    )

    raw = (response.output_text or "").strip()

    # Strip accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```$", "", raw, flags=re.IGNORECASE)

    result = json.loads(raw)

    # Normalise – ensure every speaker has a scores entry with all criteria
    scores = result.get("scores") or {}
    for speaker in speakers:
        if speaker not in scores:
            scores[speaker] = _zero_scores()
        else:
            entry = scores[speaker]
            for k in CRITERIA_KEYS:
                if k not in entry:
                    entry[k] = 0
                else:
                    entry[k] = max(0, min(10, int(entry[k])))

    result["scores"] = scores
    return result


def classify_all_threads(
    client: OpenAI,
    model: str,
    threads: list[dict],
    *,
    verbose: bool = False,
) -> list[dict]:
    """Classify every thread; returns enriched thread list."""
    enriched = []
    for i, thread in enumerate(threads):
        if verbose:
            print(f"  classifying {thread.get('thread_id','?')} ({i+1}/{len(threads)})…", flush=True)
        try:
            result = classify_thread(client, model, thread)
        except Exception as exc:
            print(f"    warning: {thread.get('thread_id')} – {exc}")
            result = {"topic": "unknown", "topic_summary": "", "scores": {}}

        enriched.append({
            **thread,
            "topic":         result.get("topic", thread.get("topic", "general")),
            "topic_summary": result.get("topic_summary", ""),
            "llm_scores":    result.get("scores", {}),
        })
    return enriched


def aggregate_person_scores(enriched_threads: list[dict]) -> dict[str, dict[str, int]]:
    """Sum LLM criterion scores across all threads per person."""
    totals: dict[str, dict[str, int]] = {}
    for thread in enriched_threads:
        for person, scores in (thread.get("llm_scores") or {}).items():
            if person not in totals:
                totals[person] = _zero_scores()
            for k in CRITERIA_KEYS:
                totals[person][k] += int(scores.get(k) or 0)
    return totals
