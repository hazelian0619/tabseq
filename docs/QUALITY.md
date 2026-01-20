# Quality Guidelines (Optional, Recommended)

This document is intentionally separate from `docs/EXECUTION.md`. The execution doc focuses on getting an end-to-end pipeline running; this file covers quality practices that keep iteration safe and reproducible.

## Minimal Test Targets

- Label encoding is correct:
  - `encode()` returns a `leaf_idx` in `[0, n_bins-1]`
  - `decode_sequence(encode(y))` returns a value inside the corresponding bin interval
  - `encode_multi_hot()` has shape `[depth, n_bins]` and the number of `1`s per row halves each step until 1
- Data pipeline is stable:
  - `TabSeqDataset.__getitem__()` returns required keys and correct tensor dtypes/shapes
- Training smoke:
  - one forward + loss + backward pass runs on a tiny batch without error

## Tooling (Planned)

- Test runner: `pytest`
- Suggested layout: `tests/test_*.py`

## How To Run

Once tests exist:

```bash
python -m pytest -q
```

