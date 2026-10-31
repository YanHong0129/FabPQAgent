# Fab PQ Agent

A product-quality troubleshooting agent for semiconductor fab data. Ask it things like
*"Lot 2 failed, which sensors look unusual?"* or *"Is s059 in statistical control?"*. A
tool-calling investigator gathers evidence (SQL, a fail classifier, SPC checks, troubleshooting
notes), then a writer turns it into a short report for a process engineer.

I built it to learn how ML and LLM agents apply to yield and quality analysis in semiconductor
manufacturing, end to end: a predictive model for failing lots, classic SPC, retrieval over
troubleshooting notes, a tool-calling agent, and an evaluation of the agent's cost, latency and accuracy.

**Data:** [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom): 1,567 production lots,
590 anonymised sensors, 104 fails (6.6%). The troubleshooting notes in `data/notes/` are
**synthetic** (SECOM's sensors are anonymised); they reference the sensors the model finds important.

## Setup
```sh
uv sync
cp .env.example .env              # add OPENROUTER_API_KEY (free models are enough)
uv run python -m fabpqagent.data  # download SECOM -> data/secom.db
uv run pytest                     # offline tests (no API calls)
```

## Who wrote what
| File | Status |
|---|---|
| `data.py`, `db.py`, `llm.py`, `memory.py`, `app.py`, tests | provided scaffold |
| `model.py`, `spc.py` | **Day 1, you** |
| `retrieval.py`, `tools.py`, `agent.py` | **Day 2, you** |
| `evaluate.py`, `eval/golden.jsonl` answers, this README's Results | **Day 3, you** |

Each stub's docstring is the spec; its test file is the definition of done.

## 3-day plan

### Day 1: model + SPC (`uv run pytest tests/test_model.py tests/test_spc.py`)
- [ ] Explore in a notebook/REPL: fail rate, missing data per sensor, fails over time
- [ ] `select_features`, `train` (LogisticRegression first, then RandomForest; keep the better PR-AUC)
- [ ] Choose a threshold deliberately (precision/recall curve) and write down why
- [ ] `predict_fail`, `top_features`, `save`/`load`; `python -m fabpqagent.model` saves `data/model.joblib`
- [ ] `spc_check` (rule 1 + rule of eight)
- [ ] Fill `must_mention` in `eval/golden.jsonl` from facts you verify yourself with SQL

### Day 2: agent (`uv run pytest tests/test_retrieval.py tests/test_tools.py tests/test_agent.py`)
- [ ] BM25 `NoteIndex` (from memory, not the CS329Z code)
- [ ] Four more tool schemas + `execute` with errors-as-observations
- [ ] `investigate`: investigator loop with budgets, writer call, memory
- [ ] Write the two prompts; try 3 questions live in `uv run streamlit run app.py`

### Day 3: evaluate + polish (`uv run pytest tests/test_evaluate.py`)
- [ ] `evaluate.py`, run it, paste the summary table below
- [ ] Look at the worst 2 questions; fix a prompt or tool; re-run; note what changed
- [ ] Screenshot the Streamlit app into `docs/`, fill in Results and Design decisions
- [ ] Push to GitHub (check `.env` is not committed)

## Results
_TODO Day 3._ Model: test PR-AUC __ vs baseline 0.066; precision __ / recall __ at threshold __.
Agent on 10 golden questions: mention recall __, expected-tool rate __, median latency __s, total tokens __.

## Design decisions
- Why PR-AUC, and how the threshold was chosen: only 6.6% of lots fail, so "always pass" scores
  93.4% accuracy while catching nothing. PR-AUC measures how well fails are ranked above passes,
  and its baseline is the fail rate (0.066), so the random forest's 0.186 reads as ~2.8x better than
  random (logistic regression: 0.149). The threshold is a business decision: I assumed a missed bad
  lot costs about 10 false alarms and picked the lowest-cost threshold on out-of-fold training
  predictions, never on the test set. Any ratio from 5 to 15 lands at ~0.09, so the choice is not
  sensitive to that guess. At 0.09 the test set gives precision 0.17 / recall 0.46 (12 of 26 fails,
  ~2.6x the hit rate of random inspection). The test set has only 26 fails, so these numbers are
  noisy, and the forest's probabilities are not calibrated (it rarely scores above 0.3).
- Why BM25 over embeddings for these notes: the notes are short and full of sensor IDs. An
  engineer searching for s519 needs the s519 note, not the s510 one, and embeddings tend to treat
  near-identical IDs as similar. BM25 matches exact tokens, needs no model or API call, and every
  score can be explained term by term. The catch with only 12 notes: IDF comes from this tiny
  corpus, so "should" looks as rare as "s510" (IDF 2.16 each), and the full q07 sentence ranks the
  escalation note above the s510 note. I kept BM25 plain and have the `search_notes` tool ask the
  LLM for keyword queries ("s510 high"); the eval checks whether that holds. A larger system would
  use hybrid search (BM25 + embeddings).
- Why two roles (investigator + writer) instead of one agent: the investigator's context fills
  with raw tool output (CSV rows, SPC lot lists) and is re-read on every turn. The writer gets only
  its prompt, the question and the evidence, so its call is short and cheap, and with no tool
  schemas passed it cannot call tools or wander off. It always runs, even after a turn or token
  budget stop; it is then told the evidence may be incomplete so it says so instead of guessing.
- How tool errors are handled and why: `execute` never raises. An unknown tool, invalid JSON,
  a wrong argument type, a lot that does not exist, non-SELECT SQL, or any unexpected exception
  comes back as an `Error: ...` observation with a specific hint (e.g. the valid lot_id range), so
  the model can fix its call on the next turn and one bad call never crashes the investigation.
  SQL also runs on a read-only connection as a second line of defence. Observations are capped at
  2,000 characters with a truncation notice, because every token is re-read on every later turn.
- Free-model fallback and what it costs in latency:
- What the eval does and does not measure:
