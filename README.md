# llm-docs-rag

CLI RAG over the Angular docs: ask a natural-language question, retrieve relevant sections, generate an answer with citations, then measure quality with a 30–50 question eval set.

## Goal

```
> how do I create a component?
[retrieved chunks…]
💬 Angular components are marked with @Component…
📎 Sources: guide/components §…, …
```

The eval set is what turns this from a demo into something you can score and improve.

## Corpus

| File | Role |
| --- | --- |
| `data/llms.txt` | Index of links (~109 lines) — not the full docs |
| `data/llms-full.txt` | Full concatenated docs (~12k lines / ~750KB) — **use this** |

Stage 1 chunks `llms-full.txt` directly. No fetch step needed.

**Stage 0 (do this first):** open `llms-full.txt` and note structure — Markdown headers, HTML-ish wrappers, code blocks. Chunking depends on what you see (e.g. split on `##` / `###` if those are reliable section boundaries).

## Pipeline

Build and verify each stage before moving on. Do not wire the whole stack and debug at the end.

| Stage | What | Done when |
| --- | --- | --- |
| **0** Inspect | Open the corpus; note size and structure | You know how sections are delimited |
| **1** Load & chunk | Read file → split (headers or ~500–1000 chars + overlap) | Chunk count printed; 3–4 chunks look coherent by eye |
| **2** Embed & store | Embed chunks → Chroma (local) | DB vector count matches chunk count |
| **3** Retrieve | Embed a question → top-k (start 3–5) | Retrieved chunks are actually relevant |
| **4** Generate | Prompt with retrieved context only | Answers match what you know about Angular |
| **5** Citations | Tag chunks; model cites sources | Spot-check: citations support the claims |
| **6** Eval | 30–50 Qs with known answers + “not in docs” cases | Scored run you can improve against |

**Embeddings (Stage 2):** OpenAI `text-embedding-3-small` or a local model — pick one and stick with it for v1.

**Retrieval (Stage 3):** Obsess here. Bad retrieval cannot be fixed by prompting.

**Eval mix (Stage 6):** factual, how-to, and questions the docs do *not* cover (must refuse / say “not in docs”).

## Next session

Stages **0** and **1** only:

1. Inspect `data/llms-full.txt`
2. Chunk it
3. Print chunk count + a few samples
4. Eyeball quality — nothing else

Prove a clean list of sensible chunks before touching embeddings.

## Shortcut checklist

```
Load llms-full.txt
→ Chunk
→ Embed + store in Chroma
→ Retrieve top-k
→ Generate with citations
→ Eval set (30–50 questions)
```
