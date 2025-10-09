import numpy as np
from pathlib import Path
from src.evaluation.cnn.eval_map import eval_and_save
from src.data.loading import load_data_map

if __name__ == "__main__":
    _, _, _, _, X_test, y_test = load_data_map()
    model_path = sorted((Path("artifacts/cnn/map")).glob("**/best.keras"))[-1]
    outdir = model_path.parent / "eval"
    eval_and_save(model_path, X_test, y_test, outdir)
