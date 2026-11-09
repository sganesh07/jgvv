import json
import os
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


SYSTEM_MARKERS = (
    "created the group",
    "joined via invite link",
    "Messages and calls are end-to-end encrypted",
    "made You an admin",
    "made an admin",
    "added",
    "group-history message",
)

TOPIC_KEYWORDS = OrderedDict(
    [
        ("school_reunion", ["school", "jgvv", "teacher", "lkg", "8th", "10th", "class photo", "khokho", "naidu", "vijayadasami"]),
        ("identity_intro", ["hello", "intro", "this is", "i am", "i'm", "nice to meet", "long time"]),
        ("photo_memory", ["photo", "pic", "image", "remember", "faces", "names", "memory", "familiar"]),
        ("work_life", ["retire", "job", "flipkart", "ai", "education", "bengaluru", "non profit", "work", "career"]),
        ("life_update", ["based now", "in germany", "retired", "settled", "traveling", "based in", "living"]),
    ]
)


def parse_ts(raw: str):
    if not raw:
        return None
    raw = raw.strip()
    for fmt in (
        "%m/%d/%y, %I:%M:%S %p",
        "%m/%d/%y, %I:%M %p",
        "%m/%d/%y, %H:%M:%S",
        "%m/%d/%y, %H:%M",
    ):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    return None


def clean_text(value: str) -> str:
    text = (value or "").strip()
    text = text.replace("<image omitted>", "")
    text = text.replace("<GIF omitted>", "")
    text = text.replace("[Image]", "")
    text = text.replace("[GIF]", "")
    text = text.replace("[group-history message]", "")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_person_and_text(raw_body: str):
    body = (raw_body or "").strip()
    if not body:
        return "system", ""
    if body.startswith("-"):
        body = body[1:].strip()
    if ": " in body and not body.lower().startswith("http"):
        person, text = body.split(": ", 1)
        person = person.strip()
        text = clean_text(text)
        return person, text
    return "system", clean_text(body)


def load_chat_messages(path: str):
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Chat export not found: {path}")

    messages = []
    current_date = None
    for raw_line in path_obj.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# ") or line.startswith("---") or line.startswith("Export date:"):
            continue
        if line.startswith("## "):
            current_date = line[3:].strip()
            continue

        match = re.match(r"^\[(?P<ts>[^\]]+)\]\s*(?P<body>.*)$", line)
        if not match:
            continue

        ts = match.group("ts")
        body = match.group("body").strip()
        person, text = extract_person_and_text(body)
        text = clean_text(text)
        lower_text = text.lower()
        msg_type = "text"
        if not text:
            msg_type = "system"
        elif any(marker in lower_text for marker in SYSTEM_MARKERS):
            msg_type = "system"
        elif "image omitted" in lower_text or "[image" in lower_text or "<image" in lower_text or "gif" in lower_text:
            msg_type = "media"
        elif person == "system":
            msg_type = "system"

        messages.append(
            {
                "id": f"msg-{len(messages) + 1}",
                "date": current_date,
                "timestamp": parse_ts(ts),
                "timestamp_iso": parse_ts(ts).isoformat() if parse_ts(ts) else None,
                "author": person,
                "text": text,
                "summary": text[:180] if text else "System notice",
                "type": msg_type,
            }
        )

    return messages


def names_in_text(text: str):
    if not text:
        return set()
    words = re.findall(r"[A-Z][A-Za-z0-9 .]+(?: JGVV| India No| No)?|You|[A-Z][a-z]+", text)
    if not words:
        return set()
    stopwords = {"Hi", "Hello", "This", "That", "Nice", "Great", "I", "We", "The", "You", "Are", "How", "Where", "Yes", "No", "Not", "But", "And", "All", "For", "From", "With", "Still", "Now", "After", "Before", "Let", "Good", "Day"}
    names = set()
    for item in words:
        clean = item.strip().strip(".,;:!?()[]{}\"'")
        if clean and clean not in stopwords and len(clean) > 2:
            names.add(clean)
    return names


def topic_for_messages(messages):
    if not messages:
        return "system"
    text_blob = " ".join(item["text"] for item in messages if item.get("text"))
    lower = text_blob.lower()
    for topic_name, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            return topic_name
    if len(messages) <= 2:
        return "introduction"
    return "general_catch_up"


def _thread_should_continue(current_thread, message, last_message):
    if not current_thread["messages"]:
        return True

    gap_seconds = 0
    if last_message.get("timestamp") and message.get("timestamp"):
        gap_seconds = (message["timestamp"] - last_message["timestamp"]).total_seconds()

    current_people = {m["author"] for m in current_thread["messages"] if m.get("author") and m["author"] != "system"}
    author_overlap = message["author"] in current_people
    name_mentions = bool(current_people.intersection(names_in_text(message["text"])))
    same_topic = topic_for_messages(current_thread["messages"]) == topic_for_messages(current_thread["messages"] + [message])

    if gap_seconds <= 1800 and (author_overlap or name_mentions or same_topic):
        return True
    if gap_seconds <= 5400 and (author_overlap or name_mentions):
        return True
    if gap_seconds <= 3600 and same_topic:
        return True
    return False


def build_threads(messages):
    threads = []
    current = None

    for message in messages:
        if not message.get("text") and message.get("type") == "system":
            if current is not None and current["messages"] and current["messages"][-1].get("timestamp") and message.get("timestamp"):
                gap = (message["timestamp"] - current["messages"][-1]["timestamp"]).total_seconds()
                if gap <= 7200:
                    current["messages"].append(message)
                    continue
            threads.append(current) if current else None
            current = {
                "thread_id": f"thread-{len(threads) + 1}",
                "people": ["system"],
                "topic": "system_notice",
                "messages": [message],
            }
            continue

        if current is None:
            current = {
                "thread_id": f"thread-{len(threads) + 1}",
                "people": [],
                "topic": "",
                "messages": [],
            }

        if not current["messages"]:
            current["messages"].append(message)
        else:
            last_message = current["messages"][-1]
            if _thread_should_continue(current, message, last_message):
                current["messages"].append(message)
            else:
                threads.append(current)
                current = {
                    "thread_id": f"thread-{len(threads) + 1}",
                    "people": [],
                    "topic": "",
                    "messages": [message],
                }

        people_set = {m["author"] for m in current["messages"] if m.get("author") and m["author"] != "system"}
        current["people"] = sorted(people_set)
        current["topic"] = topic_for_messages(current["messages"])

    if current is not None:
        threads.append(current)

    result = []
    for index, thread in enumerate(threads, start=1):
        result.append(
            {
                "thread_id": thread["thread_id"],
                "topic": thread["topic"],
                "people": thread["people"],
                "messages": thread["messages"],
                "message_count": len(thread["messages"]),
                "first_timestamp": thread["messages"][0].get("timestamp_iso") if thread["messages"] else None,
                "last_timestamp": thread["messages"][-1].get("timestamp_iso") if thread["messages"] else None,
            }
        )

    return result


def write_digest(path: str, messages):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    digest = build_threads(messages)
    out_path.write_text(json.dumps(digest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return digest


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "input" / "oct28" / "chat.txt"
    output_path = base_dir / "input" / "oct28" / "thread_digest.json"
    messages = load_chat_messages(str(input_path))
    digest = write_digest(str(output_path), messages)
    print(f"Loaded {len(messages)} messages and produced {len(digest)} threads.")
    for thread in digest[:5]:
        print(f"{thread['thread_id']}: {thread['topic']} | people={thread['people']} | count={thread['message_count']}")
