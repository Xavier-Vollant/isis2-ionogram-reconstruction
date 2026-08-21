"""Small image-loading helpers shared by the scan pipeline and notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as a grayscale floating-point array."""
    with Image.open(path) as handle:
        return np.asarray(handle.convert("L"), dtype=float)
