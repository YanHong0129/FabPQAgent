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


| File                                                              | Status            |
| ----------------------------------------------------------------- | ----------------- |
| `data.py`, `db.py`, `llm.py`, `memory.py`, `app.py`, tests        | provided scaffold |
| `model.py`, `spc.py`                                              | **Day 1, you**    |
| `retrieval.py`, `tools.py`, `agent.py`                            | **Day 2, you**    |
| `evaluate.py`, `eval/golden.jsonl` answers, this README's Results | **Day 3, you**    |

Each stub's docstring is the spec; its test file is the definition of done.

## 3-day plan

### Day 1: model + SPC (`uv run pytest tests/test_model.py tests/test_spc.py`)

- [X]  Explore in a notebook/REPL: fail rate, missing data per sensor, fails over time
- [X]  `select_features`, `train` (LogisticRegression first, then RandomForest; keep the better PR-AUC)
- [X]  Choose a threshold deliberately (precision/recall curve) and write down why
- [X]  `predict_fail`, `top_features`, `save`/`load`; `python -m fabpqagent.model` saves `data/model.joblib`
- [X]  `spc_check` (rule 1 + rule of eight)
- [X]  Fill `must_mention` in `eval/golden.jsonl` from facts you verify yourself with SQL

### Day 2: agent (`uv run pytest tests/test_retrieval.py tests/test_tools.py tests/test_agent.py`)

- [X]  BM25 `NoteIndex` (from memory, not the CS329Z code)
- [X]  Four more tool schemas + `execute` with errors-as-observations
- [X]  `investigate`: investigator loop with budgets, writer call, memory
- [X]  Write the two prompts; try 3 questions live in `uv run streamlit run app.py`

### Day 3: evaluate + polish (`uv run pytest tests/test_evaluate.py`)

- [X]  `evaluate.py`, run it, paste the summary table below
- [X]  Look at the worst 2 questions; fix a prompt or tool; re-run; note what changed
- [ ]  Screenshot the Streamlit app into `docs/`, fill in Results and Design decisions
- [ ]  Push to GitHub (check `.env` is not committed)

## Results

**Model:** test PR-AUC 0.186 vs baseline 0.066 (logistic regression: 0.149); precision 0.17 /
recall 0.46 at threshold 0.09.

**Agent on 10 golden questions** (free OpenRouter models, `uv run python -m fabpqagent.evaluate`):

| Run | What changed | Mention recall | Expected tool | Median latency | Total tokens |
| --- | --- | --- | --- | --- | --- |
| 1 | first prompts | 0.65 | 0.9 | 11.1 s | 26,433 |
| 1, rescored | same reports; metric accepts equivalent wordings (`a\|b`) | 0.80 | 0.9 | 11.1 s | 26,433 |
| 2 | prompts: unusual = \|z\| ≥ 2, trends by week/month, when to claim a cause; retry on 429 | 0.80 | 1.0 | 9.2 s | 38,742 |
| 3 | q09 rewritten (it was ambiguous) | 0.85 | 1.0 | 8.7 s | 45,582 |
| 4 | nothing (repeat of run 3) | 0.85 | 1.0 | 11.4 s | 45,138 |

Run 1's q10 failed on a rate limit (scored 0, no tokens counted), which is why its total is lowest.

- **The metric moved more than the agent at first.** Reading the reports showed two correct answers
  scored as misses (q02 said "week 33", q04 said "out of statistical control") and one wrong answer
  that scored 0.5 (the old q09 called s316, z ≈ 0, the sensor "in common" and mentioned s028 in
  passing). Rescoring the same reports moved recall from 0.65 to 0.80 with no agent change.
- **Prompt rules changed behaviour more than the score.** After run 2, q10 finished in 6 calls and
  12.7k tokens instead of 8 calls, 47k tokens and `max_turns` (an earlier live run), stopped naming
  an out-of-control sensor as the cause, and q09 stopped calling a normal sensor "common".
- **Noise floor:** runs 3 and 4 scored the same on every question, but q10's answers differed: run 3
  named s510 as the likely cause from a single lot, run 4 said the cause was not determined and hit
  `max_turns`. The score looked stable; the agent was not.
- **Still failing:** q07 drops the note's calibration advice in every run (identical text, so a
  stable writer omission, not noise). q10 compares October (6.1%) with September (2.9%) instead of
  the overall 6.6%, so "the fail rate has increased" is misleading: October is below average.

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
- Free-model fallback and what it costs in latency: `OpenRouterLLM` tries gemma, then nemotron.
  Free tiers fail often (429 per-minute token caps, 502s), so fallback keeps the agent answering,
  but it costs time: in run 1, q09 fell back to nemotron and took 103 s against a median of 11 s,
  and q10 took 140 s in run 3 against 23 s in run 4. If every model is rate-limited, the client
  waits 60 s and retries once; other errors (bad key, 5xx) are not retried, because waiting would
  not fix them. nemotron is a reasoning model that can spend all 1,200 max tokens thinking and
  return no text, which once left the report empty; an empty reply now counts as a failure and
  falls back. Tokens and time spent on failed attempts are not in the agent's totals, so the real
  cost is higher than the eval reports. Median latency is used because these outliers would
  dominate a mean.
- What the eval does and does not measure: mention recall is case-insensitive string matching
  of hand-checked facts (`a|b` accepts equivalent wordings), so it is cheap, deterministic and
  explainable, unlike an LLM judge. It checks that key facts appear, not that the conclusion is
  right: it cannot read negation ("not out of control" contains "out of control"), and a report
  that lists every candidate can score well. That is why I read the reports, not just the score.
  Expected-tool rate checks tool choice, not how the result was read (old q09 called the right
  tool and drew the wrong conclusion). A question must have one clear answer: q09 asked which
  sensor the last five fails had "in common", which had three readings, so I rewrote it with an
  explicit rule and added must_mention "1519" so that guessing "none" without finding the lots
  scores only half. With 10 questions each is worth 0.1, and two identical runs are a small sample
  for the noise floor.
