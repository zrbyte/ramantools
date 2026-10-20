# AGENTS
All code changes in this testing suite must include inline comments explaining purpose and behaviour.

`notebook_output_checks.ipynb` is generated: change `make_notebook_output_checks.py` and rerun it (`test_notebook_outputs.py` fails when the two drift apart). Every code cell needs an `expect` metadata block; the generator's docstring lists the keys.
