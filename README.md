# Angular docs RAG — eval write-up

A retrieval-augmented CLI over Angular’s concatenated docs (`data/llms-full.txt`, ~17k lines). Same index and prompt, two generators: GPT-4.1-mini (OpenAI) and Qwen3 (Ollama, local). The question was not “can it answer” — it was whether retrieval and generation fail for different reasons, and whether an LLM-judge earns its cost on a mostly factual eval set.

The short answer to both: retrieval is the stable, measurable failure mode, and almost everything else on this eval moves between runs by more than the effects I was trying to measure.

## Architecture

Chunking splits on `##` / `###`, then packs paragraphs to ~1k characters with previous-paragraph overlap inside a section. Blind character splits broke code fences and orphaned headings; header-first keeps a chunk about one topic. Embeddings are OpenAI `text-embedding-3-small` into local Chroma, upserted by a stable id (`corpus-heading-section-part`) so re-ingests update rather than duplicate. Retrieval is dense top-5, optionally filtered to `doc_version=current`. Generation is grounded in those excerpts only and must cite chunk ids.

Tradeoff: one embedding model for both stacks keeps the comparison honest (retrieval is not the variable). The cost is that Qwen never gets a local embedder, and dense-only search is weak on “what class is X” questions.

## Eval

Eighty questions: factual API names, howtos, and ten “not in the docs” abstains. **Retrieval and generation are scored separately.** Context recall is whether every `must_include` anchor appears in the retrieved text — no extra LLM. Generation is keyword match (and abstain-phrase match) on the answer. A miss with the keyword already in context is a generation failure; a miss with the keyword absent is retrieval. That split is the whole point: those need opposite fixes.

Keywords are a cheap proxy. They punish valid paraphrase (`ng g c` vs `ng generate component`) and reward a token used wrongly (`model` on a native `<input>`). For howtos I added an LLM-judge (question + answer + `expect_abstain` only — no keyword list, no chunks), temperature 0, JSON `{passed, reason}`. I validated it on 15 frozen human labels (13 real answers from one run, 2 synthetic probes).

| Grader | vs human | κ | Cost (15 items) |
| --- | --- | --- | --- |
| Keywords | 13/15 | 0.722 | $0 |
| Judge, GPT-4.1-mini | 13/15 | 0.722 | $0.0042 |
| Judge, Qwen3 local | 12/15 | 0.571 | $0 |

The aggregate ties hide the probes, which are the only cases built to expose keyword failure:

| Probe | Human | Keyword | Judge |
| --- | --- | --- | --- |
| Valid `ng g c` paraphrase | pass | fail | pass |
| Token `model`, wrong API | fail | pass | fail |

The judge wins both keyword-failure modes, then gives the two points back on `medium-013` and `hard-037`. On this gold set it does **not** demonstrably earn its cost, and the local judge now scores *below* the free keyword baseline. Headline 80-question scores stay on keywords.

The Qwen judge also did not reproduce: it scored 13/15 on an earlier run and 12/15 here, at temperature 0, newly flipping `hard-005`. A grader that moves against frozen answers and frozen labels is not yet a measuring instrument. The GPT judge did reproduce exactly, including both disagreements.

## Results

Same Chroma, same k, same prompt; the generator is the only variable. Retrieval hit rate **0.90** on both stacks (63/70 scored; 10 abstains have no anchors), mean context recall 0.90, and on the seven misses recall is **0.0** — not near-misses. The seven miss ids are identical across every run in this write-up, which is the check that the model swap stayed clean.

| | GPT-4.1-mini | Qwen3 (local) |
| --- | --- | --- |
| Keyword score | 69/80 (86%) | 68/80 (85%) |
| Retrieval hit / recall / recall-on-miss | 0.90 / 0.90 / 0.00 | identical |
| Generation given a retrieval hit | 0.984 | 0.984 |
| Generation cost, 80 questions | **$0.0615** (~$0.0008 each) | **$0** |
| Tokens in / out | 73,455 / 20,092 | 74,239 / 46,220 |
| Latency mean / p50 / p95 | 5.8s / 5.6s / 9.6s | 89.8s / 89.4s / 141.5s |
| Wall clock, 8 workers | **69.6s** | **933.7s** |

Two caveats on those last two rows, because they are easy to overstate. Qwen’s per-call latency is inflated by queueing: eight workers against one local Ollama instance mostly wait, and the same model answering a single uncontended question took **28.8s**. The defensible comparison is throughput, 13.4x, not the per-call means. And `$0` covers generation only — retrieval still bills OpenAI for embeddings on both stacks, and I do not instrument that, so the local stack is cheaper here but not free end to end.

Qwen’s 2.3x output tokens are most of its latency: it is a reasoning model and narrates more for the same graded content.

### The comparison is smaller than the noise

An earlier pair of runs scored GPT 72/80 and Qwen 69/80, and an earlier draft of this document spent a section explaining that three-point gap case by case. Rerunning GPT against the same index, prompt, and questions produced **69/80**. The model moved three points against itself, flipping `abstain-010`, `easy-007`, and `hard-033`. The gap I had explained was the same size as one model’s variance, so the per-question story was reading signal into sampling noise.

On the current pair the stacks are one point apart, and even that hides more churn than it shows — seven individual questions disagree:

| Passed only by | Ids | Why it is not a capability difference |
| --- | --- | --- |
| GPT | `abstain-003`, `abstain-006`, `abstain-008` | Both refuse; only GPT’s phrasing is in the abstain-hint list. Scorer, not model. |
| GPT | `hard-013` | Qwen generation miss — the one case per run where the anchor was in context and unused. |
| Qwen | `medium-009`, `medium-010` | **Retrieval missed for both.** Qwen emitted the anchor from parametric knowledge and the keyword scorer rewarded it. |
| Qwen | `hard-033` | GPT generation miss this run; GPT passed it last run. |

`medium-009` / `medium-010` are the reversal worth noting. The earlier draft used `easy-007` to argue that GPT’s lead was partly a parametric leak rewarded by keywords. That is still the right mechanism, but this run it is Qwen doing the leaking and GPT failing the same item. The lesson survives; the attribution to a particular model did not.

What did not move across any run: the seven retrieval misses, at recall 0.0, always the same ids. **That is the finding with a real effect size, and it is the backlog.**

## What is wrong with this eval

The scores are single runs of a stochastic system. With n=80 and observed self-variance of ±3, no difference of this size between two stacks is interpretable. I report the pair as comparable and do not rank them.

Keyword matching false-positives on short anchors (`signal`, `on` / `when`) and on any answer that names the token while explaining the wrong thing (`medium-009` Qwen; `synth-false-keyword-013`). It also false-negatives correct paraphrase.

Context recall is “did the string appear anywhere in top-k,” not “did we retrieve the gold chunk.” It overestimates retrieval quality: a stray mention of `HttpClient` in an interceptor chunk would count as a hit. I never labeled gold chunk ids for the 80, so I cannot report true recall@k.

Abstention is a closed set of English snippets, so it fails real refusals over phrasing. Six of the ten abstains failed for Qwen and three for GPT while both were correctly declining to answer. This is the scorer, and it is the largest single source of apparent difference between the stacks.

The judge called `model()` non-existent — it is in the corpus — and failed a correct howto (`medium-013`). GPT and Qwen made the same error in every run: priors overrode the domain, so it is a systematic blind spot on recent APIs, not noise. It also passed stub-components on `hard-037`, where the question asks how to *ignore unknown elements* (`NO_ERRORS_SCHEMA`). I kept the human fail; a second rater could disagree. **Single-labeler gold cannot catch my own rigidity.** I did not use the judge to score the 80, and I should not, least of all on recent-API howtos.

Generation-given-hit of 0.984 is only as good as the keyword proxy. Split scoring still cannot see a fluent wrong answer that happens to contain the token.

Cost and latency are now recorded per call in the run JSON, but latency is measured under concurrency, so it is a throughput number wearing a latency label. Embedding cost is not instrumented at all.

## Next

Repeat runs with variance bars before comparing anything again — one run per stack cannot support a claim at this effect size. Gold chunk-id recall on every question, replacing substring-in-top-k. Hybrid retrieval / reranking measured against that, since dense-only is what produces the seven stable misses. Abstention scored by the judge rather than a phrase list. Faithfulness via atomic claims vs retrieved text — correctness on the 80 is still unmeasured.

## Running it

```
cp .env.template .env      # OPENAI_API_KEY required for embeddings either way
uv sync
uv run python -m rag.store        # chunk + embed + upsert
uv run python -m rag.eval_run     # 80-question keyword + retrieval eval
uv run python -m rag.eval_judge   # judge vs eval/gold_labels.json
```

Swap stacks by uncommenting the Ollama block in `.env` (`LLM_BASE_URL=http://localhost:11434/v1`, `CHAT_MODEL=qwen3`). Set `PYTHONUNBUFFERED=1` for the local run — it takes ~15 minutes and stdout otherwise block-buffers to nothing. Prices for cost accounting live in `CHAT_PRICE_PER_MILLION` in `config/settings.py`; anything not listed, including local models, counts as $0.
