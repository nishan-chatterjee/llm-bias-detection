# Localized chat diagnostics

This directory contains the paper-facing exploratory analysis of the Political
Compass chat traces. Primary summaries are restricted to:

- Gemma 3 1B, 4B, 12B, and 27B;
- Qwen3 4B, 8B, 14B, and 32B under both no-think and think protocols.

That is eight checkpoints and 12 checkpoint–protocol variants. The
abliterated Gemma 27B checkpoint appears only in the explicitly secondary,
matched ablation notebook. GaMS and all LLM-judge annotations are excluded.

## Qualifying-expression screen

The historical code field is named `hedging`, but the paper-safe measurement is
**incidence of predefined qualifying expressions**. `feature_definitions.py`
matches strings such as “may,” “might,” “perhaps,” “it depends,” “not
necessarily,” and “to some extent.” The screen is transparent and reproducible,
but it can miss contextual hedges and count non-hedging uses. It therefore does
not establish pragmatic hedging, uncertainty, caution, or safety motivation.
Those claims would require contextual human annotation, which was not available
for this study and is listed as future work.

This limitation is consistent with the broader hedge-detection literature,
which treats the task as contextual sequence/classification modeling rather
than simple keyword lookup; see Katerenchuk and Levitan (2024), [“You should
probably read this”: Hedge Detection in Text](https://arxiv.org/abs/2405.13319v1).

For each proposition the release reports:

1. the overall cue incidence across all 12 primary variants and personas;
2. matched left/right cue incidence for the same model, LHS row, and question;
3. a right-minus-left risk difference;
4. whether the cell has at least 30 cue events across 600 answers; and
5. whether the absolute difference crosses a five-percentage-point exploratory
   screen.

The event and effect-size cutoffs define a discovery pocket; they are not a
multiple-comparison-adjusted significance claim. Recurrence counts ask whether
the same proposition/persona direction reappears in multiple checkpoints or
families. The defensible conclusion is proposition-dependent heterogeneity, not
a uniform “left hedges more” or “right hedges more” effect.

## Moral-distancing cue screen

The rule-based `moral_distance` field matches explicit non-endorsement or moral
disapproval expressions such as “I do not condone,” “I do not endorse,”
“deeply problematic,” “harmful viewpoint,” and “this position is dangerous.”
It records those strings only. It does not infer a model’s moral beliefs or the
cause of the language.

## Qwen rescue and harm

Think and no-think traces are paired on Qwen size, persona, Latin-hypercube row,
and proposition. A **rescue** means the no-think answer is outside the assigned
persona’s target quadrant and the think answer is inside it. A **harm** is the
reverse. “Both correct” and “both incorrect” are retained. This is binary
quadrant membership, not distance from a quadrant centre and not a judgment of
rationale quality.

The protocols also change their recommended decoding configuration and token
budget. The comparison is therefore a complete protocol comparison, not a pure
causal estimate of hidden reasoning.

### Can length or content distinguish rescues from harms?

The embedding diagnostic uses question-grouped 10-fold validation so that a
proposition does not occur in both train and test. Balanced rescue/harm samples
are predicted with logistic regression from length only, BGE-M3 embeddings
only, or both. For visible rationales, macro-F1 is 0.499, 0.548, and 0.549;
for hidden Qwen thinking it is 0.546, 0.564, and 0.562. These values are modest
diagnostics, not evidence that the content reliably explains why rescues occur.
The exact machine-readable values are in
`artifacts/tables/rescue_harm_prediction.json`.

## Incremental value of proposition wording/structure

`07_question_feature_models.py` uses GroupKFold by proposition ID. It compares
model/persona/protocol metadata against structured proposition features,
TF–IDF wording, and combined features on entirely held-out propositions. The
increment over the metadata-only baseline asks whether wording or structure
generalizes beyond the identity of questions seen during training. It is not a
causal estimate of wording. The results and plot are under
`artifacts/tables/question_feature_models/` and
`artifacts/figures/question_feature_incremental_r2.png`.

## Notebooks

The numbered notebooks are intentionally small and localized:

- `00`: qualifying-expression incidence and matched left/right cells;
- `01`: Qwen rescue/harm localization and recurrence;
- `02`: proposition-specific differential difficulty;
- `03`: why phrase discovery does not support a model-specific claim;
- `04`: secondary Gemma 27B standard/abliterated comparison;
- `06`: limited exact MCQ/chat item cross-check;
- `07`: primary-model expressive heterogeneity and appendix diagnostics.

The largest-cell plot in notebook `07` is an index of where the 30 largest
matched cue-rate differences occur. A bar to the right means more right-persona
answers in that model/regime contained a listed cue; a bar to the left means
more left-persona answers did. It is an appendix lookup aid, not a global-effect
figure.

## Rebuilding

The compact audited tables and figures are included. Rebuilding from raw traces
requires the private dataset, an authorized proposition file, and the locally
reconstructed scorer. Run:

```bash
bash tasks/political-compass/qualitative-analysis/run_cpu_pipeline.sh
```

Embedding steps are optional and require `sentence-transformers` plus the
configured BGE-M3 checkpoint. No judge server or judge model is used.
