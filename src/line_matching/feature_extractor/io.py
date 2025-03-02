from typing import Dict, Union
from pathlib import Path
import numpy as np
from flatten_dict import unflatten


def load_frenet_features(features_file: Union[str, Path]) -> Dict[str, np.ndarray]:
    features_file = Path(features_file)
    features = np.load(features_file, allow_pickle=True)
    features = unflatten(features, splitter="dot")

    return features
