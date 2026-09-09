# AI Student Support Assistant

**Use Case #1 — TNSDC / IBM Agentic AI Internship**
Answers college-related questions from regulations, syllabus, FAQs, and notices.

**Key Agent Capabilities:** RAG + Tools + Memory

---

## What it does

A chat-based agent for students that:

- **RAG** — retrieves relevant answers from the college's FAQs, syllabus, and
  regulation documents (`data/*.txt`) using TF-IDF similarity search, so it
  runs fully offline with no paid API key required.
- **Tools** — calls real actions instead of just retrieving text:
  - `check_attendance_eligibility` — applies the 75% attendance rule
  - `lookup_office_contact` — looks up office location/hours/email
  - `raise_support_ticket` — logs a complaint/escalation for staff follow-up
- **Memory** — remembers the last few turns of conversation (short-term) and
  persists simple facts per student across sessions (long-term), stored in
  `data/`.

## Project structure

```
ai-student-support-assistant/
├── app.py              # Streamlit chat UI
├── agent.py            # Orchestrator: routes queries to tools or RAG
├── rag_engine.py        # TF-IDF retrieval + answer generation
├── tools.py             # Callable agent tools
├── memory_store.py      # Short-term + long-term memory
├── test_agent.py         # CLI demo (no Streamlit needed)
├── requirements.txt
├── data/
│   ├── faqs.txt
│   ├── syllabus.txt
│   └── regulations.txt
└── README.md
```

## How to run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Quick CLI demo (fastest way to verify it works):
   ```bash
   python test_agent.py
   ```

3. Full chat UI:
   ```bash
   streamlit run app.py
   ```
   Then open the local URL Streamlit prints (usually http://localhost:8501).

### Optional: natural-language generation with an LLM

By default the RAG engine returns the best-matching document snippet
directly (extractive answer) — no API key needed. If you set an
`OPENAI_API_KEY` environment variable, it will instead use the OpenAI API
to phrase a more natural answer grounded in the same retrieved context:

```bash
export OPENAI_API_KEY=sk-...
streamlit run app.py
```

## How to extend

- Add more knowledge: drop additional `.txt` files into `data/` (blank-line
  separated blocks = one retrievable chunk each), then call
  `agent.rag.reload()` or restart the app.
- Add a new tool: write a function in `tools.py`, then add a trigger rule
  in `agent.py`'s `_detect_tool_intent()`.
- Swap the retrieval method: `rag_engine.py` is self-contained — TF-IDF can
  be swapped for embeddings (e.g. sentence-transformers, watsonx.ai
  embeddings) without touching `agent.py` or `app.py`.

## Submitting to GitHub

```bash
cd ai-student-support-assistant
git init
git add .
git commit -m "AI Student Support Assistant - Day 1 use case"
git branch -M main
git remote add origin https://github.com/<your-username>/ai-student-support-assistant.git
git push -u origin main
```

Then submit the repository link as instructed.
