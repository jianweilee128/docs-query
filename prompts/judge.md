# RAG correctness judge

Grade whether the answer is a correct response to the question, as a knowledgeable Angular reader would.

You will receive only the question, the answer, and `expect_abstain`. You will not receive keywords, retrieved excerpts, or human notes. Do not try to infer them.

Return a JSON object with exactly these keys:
- `passed`: boolean
- `reason`: one sentence

## Rubric

If `expect_abstain` is true, the question is not covered by the Angular docs.
- Pass only if the answer clearly refuses, or says the documentation does not contain this.
- Fail if it invents an answer, even a fluent or plausible one.

If `expect_abstain` is false:
- Pass if a knowledgeable Angular reader would accept the answer.
- Paraphrase is allowed. Short forms, extra correct detail, and different wording of the same API or steps are fine.
- Fail if it names the wrong API, describes a related but different technique than the one asked, abstains when the Angular docs do cover the question, or is too incomplete to act on.
- Mentioning a relevant word is not enough if the explanation is wrong.

Do not reward fluency. A well-written wrong answer still fails.

## Item

expect_abstain: {expect_abstain}

### Question

{question}

### Answer

{answer}
