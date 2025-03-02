import matlab
import matlab.engine
from typing import *
import tempfile
import numpy as np
from inv3d_util.load import save_image
from pathlib import Path
from singleton_decorator import singleton


@singleton
class MatlabEngine:

    def __init__(self):
        self.eng = matlab.engine.start_matlab()

        matlab_dir = str(Path(__file__).parent)
        paths = self.eng.genpath(matlab_dir)
        self.eng.addpath(paths, nargout=0)

    def score(self, norm_image: np.ndarray, true_image: np.ndarray) -> Dict:

        with tempfile.TemporaryDirectory() as tmpdirname:
            norm_image_path = Path(tmpdirname) / "norm_image.png"
            true_image_path = Path(tmpdirname) / "true_image.png"
            save_image(norm_image_path, norm_image)
            save_image(true_image_path, true_image)

            ms_gray, ld_gray = self.eng.ld_mssim(str(norm_image_path), str(true_image_path), nargout=2)

            return {
                "version": str(self.eng.version()),
                "matlab_mssim": float(ms_gray),
                "matlab_ld": float(ld_gray),
            }
