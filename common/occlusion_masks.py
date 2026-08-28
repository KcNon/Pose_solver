"""Label-selective masks for known modal occluders."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def known_occluder_mask(
    labels: np.ndarray,
    target_label: int,
    occluder_labels: Iterable[int] | None = None,
) -> np.ndarray:
    """Return labelled occluders while always excluding the target itself.

    ``None`` preserves the historical behaviour: every non-zero, non-target
    label is an occluder.  An explicit iterable allows controlled experiments
    such as rigid parts only (``[1, 2]``) versus rigid parts plus hand
    (``[1, 2, 3]``).
    """

    values = np.asarray(labels)
    if values.ndim != 2:
        raise ValueError("known occluder labels require a 2-D label map")
    target = int(target_label)
    if occluder_labels is None:
        return (values != 0) & (values != target)
    selected = sorted({int(value) for value in occluder_labels})
    if any(value < 1 or value > 255 for value in selected):
        raise ValueError("known occluder labels must be in [1, 255]")
    selected = [value for value in selected if value != target]
    if not selected:
        return np.zeros(values.shape, dtype=bool)
    return np.isin(values, np.asarray(selected, dtype=values.dtype))
