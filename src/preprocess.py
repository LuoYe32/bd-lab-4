import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import read_config, ensure_dir


class FashionPreprocessor:

    def __init__(self, config_path: str):
        self.cfg = read_config(config_path)

        self.raw_train = Path(self.cfg["DATA"]["raw_train"])
        self.raw_test = Path(self.cfg["DATA"]["raw_test"])

        self.val_size = float(self.cfg["PREPROCESS"].get("val_size", 0.1))
        self.random_state = int(self.cfg["PREPROCESS"].get("random_state", 42))
        self.normalize = self.cfg["PREPROCESS"].getboolean("normalize", True)

        self.out_dir = Path("data/processed")
        ensure_dir(self.out_dir)

    def load_fashion_csv(self, csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(csv_path)

        if "label" not in df.columns:
            raise ValueError("Expected 'label' column in CSV")

        y = df["label"].to_numpy(dtype=np.int64)
        X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)

        if X.shape[1] != 784:
            raise ValueError(f"Expected 784 pixel columns, got {X.shape[1]}")

        return X, y

    def preprocess(self):

        X_train_full, y_train_full = self.load_fashion_csv(self.raw_train)
        X_test, y_test = self.load_fashion_csv(self.raw_test)

        if self.normalize:
            X_train_full = X_train_full / 255.0
            X_test = X_test / 255.0

        stratify = y_train_full
        unique, counts = np.unique(y_train_full, return_counts=True)

        if np.min(counts) < 2:
            stratify = None

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        np.savez_compressed(self.out_dir / "train.npz", X=X_train, y=y_train)
        np.savez_compressed(self.out_dir / "val.npz", X=X_val, y=y_val)
        np.savez_compressed(self.out_dir / "test.npz", X=X_test, y=y_test)

        print("Saved:")
        print(f"- {self.out_dir / 'train.npz'}: X={X_train.shape}, y={y_train.shape}")
        print(f"- {self.out_dir / 'val.npz'}:   X={X_val.shape}, y={y_val.shape}")
        print(f"- {self.out_dir / 'test.npz'}:  X={X_test.shape}, y={y_test.shape}")


def main(config_path: str):

    preprocessor = FashionPreprocessor(config_path)
    preprocessor.preprocess()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.ini")

    args = parser.parse_args()
    main(args.config)