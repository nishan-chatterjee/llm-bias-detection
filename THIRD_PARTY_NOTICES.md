# Third-party data and model notices

The repository's MIT licence applies to the release code unless a file or
component says otherwise. It does not replace upstream terms for data, model
checkpoints, model outputs, or evaluation instruments.

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

## Model checkpoints and outputs

No checkpoint weights are redistributed. Repository identifiers and pinned
revisions are recorded solely to identify the checkpoints used. Users must
comply with each model's upstream licence and terms; generated traces may also
remain subject to those terms.

## Hate-speech sources

The public companion dataset includes model probabilities, gold labels,
identity targets, and source metadata from the collaborator's completed run.
It does not include source statements or the assembled corpus itself. The run
used 13,320 target-specific rows (1,332 per target); each configuration scored
one target's 1,332 rows. The exact assembled split, component licences,
attribution, and checksum remain documented limitations requiring final
verification. The source documentation associates the material with Yoder et
al. (CoNLL 2022).
