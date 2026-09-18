# Third-party data and model notices

The repository's MIT licence applies to executable code unless a component
says otherwise. `LICENSE-CC-BY-4.0.md` applies to the authors' original
documentation, figures, templates, selection/arrangement and analysis tables.
Neither licence replaces upstream terms for borrowed data, model checkpoints
or evaluation instruments.

## IBM Claim Stance

`tasks/sentiment/data/questions/ibm-claim-stance/ibm_test_topics.csv` is a
30-topic extract from `ibm-research/claim_stance`, pinned at revision
`ec4e2c2ec3e0c70087c67a28a7bce58b682b8109`. The upstream dataset card credits
Wikipedia and IBM (2014) and states CC BY-SA 3.0 in its licence prose. See the
component README for attribution and the required Bar-Haim et al. (2017)
citation.

## Political Compass

The Political Compass name, propositions, website outputs, and original
instrument are third-party material. Proposition files and reconstructed
scoring parameters are intentionally excluded from the Git release pending
explicit redistribution clearance. The supplied code does not grant rights to
use or redistribute that material.

The `restricted_inputs/` question directory is removed from the current public
dataset tree. Older revisions may retain it; no history erasure is claimed.
Raw chat records can also contain proposition text in statements, prompts,
responses and metadata, so directory removal alone is not complete text removal.
No redistribution permission has been established. See the dataset card for
the current public trace scope and the [official FAQ](https://www.politicalcompass.org/faq)
for upstream terms. The portable aggregate analysis download avoids raw text.

## Model checkpoints and outputs

No checkpoint weights are redistributed. Repository identifiers and pinned
revisions are recorded solely to identify the checkpoints used. Users must
comply with each model's upstream licence and terms; generated traces may also
remain subject to those terms.

## Hate-speech sources

The completed run's probabilities, labels and source metadata remain unchanged.
The newly supplied assembled corpus and prompt JSON are now included separately.
The corpus has 13,320 items, exactly 666 hate and 666 non-hate per primary target.
Within-target order, labels and source metadata align with all 19,180,800
archived predictions. Original text/prompt hashes were not recorded.

The corpus is associated with [Yoder et al. (2022)](https://aclanthology.org/2022.conll-1.3/).
Its preserved `dataset` values identify these sources:

- Gab Hate Corpus (`kennedy2020`, 4,421 rows): CC BY 4.0 according to the
  [official OSF record](https://osf.io/edua3/), verified through
  [OSF's licence API](https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/).
- HateXplain (`hatexplain`, 4,326 rows): upstream repository's
  [MIT licence](https://github.com/hate-alert/HateXplain/blob/master/LICENSE),
  Copyright (c) 2020 Punyajoy Saha. Its complete notice is retained in
  `tasks/hate-speech/data/README.md` and the HF input README.
- Civil Comments (`civilcomments`, 3,051 rows):
  [Jigsaw's source data statement](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data)
  declares CC0 for both data and underlying comment text.
- Social Bias Inference Corpus (`sbic`, 1,522 rows): CC BY 4.0 per the
  [Social Bias Frames card](https://huggingface.co/datasets/allenai/social_bias_frames),
  attributed to Sap et al. (2020).

Filtering, balancing and primary-target assignment are modifications to the
source collections. Original file order, `dataset`, `grouping` and `fold` are
preserved. No original post IDs were supplied; attribution cannot be
reconstructed per post. Preserve component licences and source attribution;
the collection's CC BY 4.0 is not a replacement for those terms or for any
independent privacy/platform rights in underlying posts.
