from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class BaselineEvalSpec:
    """
    Unified eval spec to keep baselines comparable to a TabSeq "standard run".

    The critical fields for "same mouth" (口径一致):
      - seed: controls train/val split
      - confidence: target coverage probability
      - v_min/v_max: binning boundaries (TabSeq-aligned)
      - clip_range: whether to clip y_true/y_lower/y_upper into [v_min, v_max] before computing metrics
    """

    dataset: str
    seed: int
    confidence: float
    v_min: Optional[float]
    v_max: Optional[float]
    clip_range: bool = True
    width_bins: Sequence[float] = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 100.0)
    bin_step_02: Optional[float] = None
    bin_step_04: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
