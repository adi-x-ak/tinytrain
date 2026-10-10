# Code standards

- Keep the training path explicit and readable. Avoid generic plugin systems and redundant wrappers.
- Use descriptive names, small functions, type hints, and comments explaining numerical or state semantics.
- Model changes require causal-attention and reference-gradient checks.
- Training changes require full-batch/accumulation and uninterrupted/resumed checks.
- Synchronization changes require `python scripts/verify.py` with actual Gloo processes.
- Run `ruff check .`, `ruff format --check .`, and `pytest` before committing.
- Publish measured results with exact configurations. Do not claim frontier capability or cluster performance from laptop runs.
