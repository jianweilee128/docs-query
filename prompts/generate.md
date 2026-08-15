# RAG answer prompt

Answer the question using **ONLY** the documentation excerpts below.
If the excerpts do not contain enough information, say so clearly — do not guess.

## Citations

- Tag every factual claim with the source id in brackets, e.g. `[chunk-0]`.
- Only cite ids that appear in the excerpts.
- If multiple excerpts support a claim, you may cite more than one: `[chunk-0][chunk-2]`.
- End with a short **Sources** list of the ids you used.

## Question

{query}

## Documentation excerpts

{context}
