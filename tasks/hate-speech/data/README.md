# Validated hate-speech inputs

**Sensitive-content warning:** the corpus contains identity-targeted hate speech,
slurs and offensive language. The files are for research evaluation; do not
display raw examples without a content warning.

The supplied files are installed at the runner's canonical paths:

- `prompts/english.json`: five contexts, ten instruction templates and five
  short/long persona templates with ideology insertions.
- `corpus/english.jsonl`: 13,320 records, ten primary targets, exactly 666 hate
  and 666 non-hate items per target. Original file order and all supplied fields
  are preserved, including the pre-existing `fold` field; `fold` is not used to
  select a different subset by the runner.

SHA-256:

```text
prompts/english.json f3154a073122a1993d63adf2c675dc0b3954c07d63af7a1a383d5fe57d53b94c
corpus/english.jsonl 6737b0d401b33c970cd4006a5122e31b411d814056f5b261f745bd731f07196f
```

On 2026-09-16, within-target order, gold labels and `dataset;grouping` provenance
matched all 19,180,800 archived predictions across eight checkpoints. The old
outputs did not store text or prompt hashes, so this is complete **positional
metadata alignment**, not cryptographic proof that every historical prompt and
statement byte was identical. No predictions or metrics were changed.

Run the check after downloading the raw hate outputs:

```bash
python tasks/hate-speech/analysis/verify_input_alignment.py \
  --prompts tasks/hate-speech/data/prompts/english.json \
  --corpus tasks/hate-speech/data/corpus/english.jsonl \
  --outputs data/release/data/hate_speech
```

## Source attribution and licence scope

This is a filtered/balanced multi-source research corpus associated with
[Yoder et al. (2022)](https://aclanthology.org/2022.conll-1.3/) and its
[source repository](https://github.com/michaelmilleryoder/hate_speech_identities).
The `dataset` field preserves component provenance:

| Component | Rows | Upstream terms/source |
|---|---:|---|
| `kennedy2020` (Gab Hate Corpus) | 4,421 | [Official OSF record](https://osf.io/edua3/) declares CC BY 4.0; licence verified through the OSF API |
| `hatexplain` | 4,326 | [HateXplain repository MIT licence](https://github.com/hate-alert/HateXplain/blob/master/LICENSE), Copyright (c) 2020 Punyajoy Saha; full notice below |
| `civilcomments` | 3,051 | [Jigsaw source data statement](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data): CC0, including underlying comment text |
| `sbic` | 1,522 | [Social Bias Frames dataset card](https://huggingface.co/datasets/allenai/social_bias_frames): CC BY 4.0; [Sap et al. project](https://maartensap.com/social-bias-frames/) |

The release's CC BY 4.0 applies to the authors' original templates and corpus
selection/arrangement, not a blanket relicensing of these source components.
Preserve source attribution, source notices and any applicable privacy/platform
rights. The supplied corpus lacks original post IDs, so per-post attribution
cannot be reconstructed; provenance is retained at dataset/grouping level.

## HateXplain MIT notice

MIT License

Copyright (c) 2020 Punyajoy Saha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
