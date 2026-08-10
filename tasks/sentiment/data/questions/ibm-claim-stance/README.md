# IBM Claim Stance topic extract

`ibm_test_topics.csv` contains the 30 unique `(topicText, topicTarget,
topicSentiment)` records from the upstream **test** split.

- Source: <https://huggingface.co/datasets/ibm-research/claim_stance>
- Pinned revision: `ec4e2c2ec3e0c70087c67a28a7bce58b682b8109`
- Copyright notices in the upstream card: Wikipedia and IBM (2014)
- Terms stated in the upstream card prose: CC BY-SA 3.0
- Required citation: Bar-Haim et al., *Stance Classification of
  Context-Dependent Claims*, EACL 2017, <https://aclanthology.org/E17-1024/>

The full Arrow cache is deliberately not tracked. The experiment uses only the
30 unique topic rows, not the 1,355 claim-level test records.
