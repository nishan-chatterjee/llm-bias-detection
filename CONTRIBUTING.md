# Contributing

Bug reports and reproducibility fixes are welcome through GitHub issues or pull
requests. Before opening a pull request:

1. create the analysis environment from `environment-analysis.yml`;
2. run `pytest -q` and `python -m compileall -q scripts tasks dataset tests`;
3. run `bash -n` on every changed shell script;
4. preserve the pinned model and dataset revisions unless the change explicitly
   documents a new release;
5. do not commit model weights, reconstructed Political Compass scorer files,
   raw task outputs, credentials, or third-party data without verified rights;
6. state whether a change alters prompts, sampling, candidate scoring, schemas,
   row counts, or published analysis values.

Code is currently MIT-licensed. Contributions are accepted under the same
licence. Third-party datasets, models, instruments, and generated outputs retain
their applicable upstream terms.
