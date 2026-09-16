# Validated hate-speech prompts

The supplied `english.json` is included here. The clean runner validates that
it has five contexts, ten list-valued instructions containing `{text}`, five
short and five long persona templates, and insertions for all five assigned
persona conditions.

The supplied bytes are preserved; the SHA-256 is recorded in `../README.md`.

Do **not** copy the similarly named Political Compass prompt file from the
historical anonymized package: its instruction schema is incompatible with the
hate runner.
