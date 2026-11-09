# JGVV Memory Lanes – Project Notes

## What this project does
Takes a WhatsApp group-chat export (`input/<date>/chat.txt`) and produces:
1. A **memory matrix** showing each person's contribution across 6 criteria.
2. An **intro card** per person with their actual self-introduction text.
3. A **self-contained shareable HTML** you can send over WhatsApp.

---

## Pipeline (run order)

```powershell
.\Scripts\python.exe chat_insights.py \
  --chat input/oct31/chat.txt \
  --out input/oct31/chat_insights.json \
  --thread-out input/oct31/thread_digest.json \
  --llm-classify \   # semantic thread scoring via LLM
  --llm-intro \      # LLM-extracted intro per person
  --llm-model gpt-4.1-mini
```

Requires `OPENAI_API_KEY` in environment. Falls back to heuristic if key is absent.

---

## Key files

| File | Purpose |
|---|---|
| `thread_digest.py` | Parses chat, groups messages into time-based threads. No LLM. |
| `thread_classifier.py` | **LLM classifier.** Sends each thread to OpenAI. Returns semantic topic + per-person 0-10 criterion scores. |
| `chat_insights.py` | Orchestrates pipeline. Uses `thread_classifier` scores as primary criterion source. Also calls LLM for intro extraction. Writes `chat_insights.json`. |
| `memory_lanes_oct31.html` | Live page – loads `input/oct31/chat_insights.json` via `fetch`. |
| `memory_lanes_oct31_selfcontained.html` | **Shareable single file.** JSON embedded in `<script id="embeddedData" type="application/json">`. |

---

## How criterion scores work (NEW – semantic)

1. `thread_digest.py` groups messages into ~55 threads by time gaps and topic continuity.
2. `thread_classifier.py` sends each thread to GPT-4.1-mini with this prompt shape:

   ```
   System: You analyse WhatsApp threads and return strict JSON only.
   User:   [thread text]
           → return { topic, topic_summary, scores: { person: { criterion: 0-10 } } }
   ```

3. Scores are **summed** across all threads per person → `criteria_scores` in the JSON.
4. Group leaders per criterion are whoever has the highest summed score.

**Old approach (removed):** keyword counting in `score_message_for_criteria` + `CRITERION_KEYWORDS` dict. Do not bring that back.

---

## How intro cards work (semantic)

`llm_intro_excerpt` in `chat_insights.py` collects up to 40 first-person candidate messages per person and sends them to GPT:

```
System: You extract biography-like self-intro lines. Return strict JSON only.
User:   Person name: {name}
        Task: Extract self-intro facts (background, family, location, work).
        Ignore jokes, reactions, banter.
        Return {"intro_lines":["..."]} – up to 4 lines.
        Candidate messages: ...
```

Falls back to heuristic sentence-splitter if LLM returns nothing.

---

## Updating the shareable HTML after a new run

```powershell
$s = @'
from pathlib import Path
html = Path("memory_lanes_oct31_selfcontained.html").read_text(encoding="utf-8")
json_text = Path("input/oct31/chat_insights.json").read_text(encoding="utf-8").replace("</", "<\\/")
m = '<script id="embeddedData" type="application/json">'
e = '</script>'
sp = html.find(m); ep = html.find(e, sp + len(m))
Path("memory_lanes_oct31_selfcontained.html").write_text(html[:sp+len(m)] + "\n" + json_text + "\n  " + html[ep:], encoding="utf-8")
print("done")
'@
Set-Content -Path _s.py -Value $s -Encoding UTF8 ; .\Scripts\python.exe _s.py ; Remove-Item _s.py
```

---

## 6 Criteria

| Key | Label | What it measures |
|---|---|---|
| `memory_archivist` | Memory Archivist | Recalls school memories, names, events |
| `conversation_connector` | Conversation Connector | Welcomes, links people, bridges topics |
| `story_weaver` | Story Weaver | Long personal narratives |
| `vibe_lifter` | Vibe Lifter | Humour, warmth, enthusiasm |
| `prompt_catalyst` | Prompt Catalyst | Asks questions, sparks new threads |
| `life_update_anchor` | Life Update Anchor | Current life status (job, family, location) |

---

## UI features (memory_lanes_oct31*.html)

- **Tab 1 – Memory Matrix**: sortable columns (click header), selected person floats to top, shift-click for multi-select.
- **Tab 2 – Intro Content**: LLM-extracted actual intro text per person.
