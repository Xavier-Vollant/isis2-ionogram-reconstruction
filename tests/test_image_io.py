import json

import numpy as np
from PIL import Image

from isis_research.image_io import load_image
from isis_research.labeling.dataset import scan_arrays


def test_load_image_returns_grayscale_float_array(tmp_path):
    path = tmp_path / "scan.png"
    Image.fromarray(np.array([[0, 128], [255, 64]], dtype=np.uint8)).save(path)

    result = load_image(path)

    assert result.dtype == float
    assert result.shape == (2, 2)
    assert result[0, 1] == 128.0


def test_landmark_dataset_uses_shared_image_loader(tmp_path):
    image_path = tmp_path / "scan.png"
    labels_path = tmp_path / "scan_ml_labels.json"
    Image.fromarray(np.zeros((10, 20), dtype=np.uint8)).save(image_path)
    labels_path.write_text(
        json.dumps(
            {
                "image_shape": [10, 20],
                "labels": [
                    {"class_name": "top_of_ionogram", "csa_row": 1},
                    {"class_name": "bottom_of_ionogram", "csa_row": 8},
                    {"class_name": "frequency_marker", "csa_x": 5},
                ],
                "ignore": [],
            }
        )
    )

    result = scan_arrays(
        {
            "name": "scan",
            "labels_path": labels_path,
            "film_path": image_path,
            "reel": "reel",
            "station": "station",
        }
    )

    assert result["shape"] == (10, 20)
    assert result["column"]["features"].shape[0] == 1024
