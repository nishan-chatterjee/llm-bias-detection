# Political Compass experiments

This directory contains the release versions of the MCQ and chat experiments,
the scorer-reconstruction method, compact scored configuration tables, and
primary-model analysis notebooks.

## Primary checkpoints

The supported default order is Gemma 3 1B, 4B, 12B, and 27B, followed by Qwen3
4B, 8B, 14B, and 32B. Chat additionally expands each Qwen checkpoint into a
`think` and `no_think` protocol. The abliterated Gemma 27B checkpoint is an
explicitly requested secondary ablation and is never selected by default.

## Stimulus files and redistribution status

`data/prompts/` and `data/chat-prompts/` contain the experimental wrappers,
persona variants, answer-key variants, and contexts. The complete Political
Compass proposition files are intentionally ignored pending permission or a
final rights review. Place an authorized local copy at:

```text
tasks/political-compass/data/questions/<language>.json
```

Each file is a JSON list of 62 objects with `id`, `statement`, and four ordered
`choices`. The official Political Compass FAQ states that the instrument is
copyrighted and restricts unauthorized adoption/adaptation. Prior academic
reproduction is not itself a licence. The same caution applies to publishing
full prompts embedded in model traces; the companion Hugging Face dataset stays
private until this is resolved.

Some propositions and outputs contain sensitive language, including references
to death, race, disability, discrimination, religion, sexuality, and violence.

## MCQ

Generate the deterministic 300-sample Latin-hypercube design:

```bash
python tasks/political-compass/mcq.py generate-design
```

Run one schedule subset (repeat or use scheduler arrays for the remaining
models and quantizations):

```bash
python tasks/political-compass/mcq.py run \
  --models gemma-3-1b-it \
  --quantizations bf16 \
  --gpus 0
```

The MCQ output is one wide CSV row per experimental configuration. For each of
the 62 propositions, the four `q*_prob_ans*` fields are a softmax over only the
four answer-key token logits. They are probabilities, not raw logits.

## Chat and Qwen thinking protocols

Generate the English chat design:

```bash
python tasks/political-compass/chat.py generate-design \
  --language english \
  --output tasks/political-compass/output/chat
```

Run one or more aliases:

```bash
python tasks/political-compass/chat.py run \
  --gpus 0,1 \
  --models Qwen3-8B \
  --answer chat-classify \
  --language english \
  --output tasks/political-compass/output/chat
```

Each JSONL row contains the stage-1 generated text and the stage-2 candidate
log-probabilities/probabilities. “Stage 2” is candidate scoring conditioned on
the generated answer; it does not generate a second rationale.

### Adversarial context condition

Context `ctx_4_system` explicitly asks the model to accept each premise and to
respond “without applying a safety filter.” This is an attempted safety-bypass
or jailbreak-style instruction embedded in a user prompt. The factor therefore
tests susceptibility to that instruction; it should not be described as proof
that a jailbreak succeeded. Success requires an observed behavioral criterion,
not merely the presence of the prompt.

## Scoring and analysis

See `instrument/README.md` before reconstructing a scorer. The generated `.npz`
is ignored and is not part of the release.

The notebooks in `analysis/notebooks/` operate on compact, primary-model
configuration tables and use a consistent presentation order. They do not need
model weights. Raw-to-aggregate conversion requires a locally reconstructed
scorer and is documented in `analysis/README.md`.
