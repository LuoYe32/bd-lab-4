import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.utils import read_config, ensure_dir


class FashionTrainer:

    def __init__(self, config_path: str):

        self.cfg = read_config(config_path)

        self.train_path = Path("data/processed/train.npz")
        self.val_path = Path("data/processed/val.npz")

        self.model_path = Path(self.cfg["ARTIFACTS"]["model_path"])
        self.metrics_path = Path(self.cfg["ARTIFACTS"]["metrics_path"])

        self.C = float(self.cfg["LOGREG"].get("C", 1.0))
        self.max_iter = int(self.cfg["LOGREG"].get("max_iter", 200))
        self.n_jobs = int(self.cfg["LOGREG"].get("n_jobs", -1))

    def load_npz(self, path: Path):

        data = np.load(path)
        return data["X"], data["y"]

    def train(self):

        X_train, y_train = self.load_npz(self.train_path)
        X_val, y_val = self.load_npz(self.val_path)

        model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
            solver="lbfgs",
        )

        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)

        val_acc = float(accuracy_score(y_val, val_pred))
        val_f1m = float(f1_score(y_val, val_pred, average="macro"))

        ensure_dir(self.model_path.parent)

        joblib.dump(model, self.model_path)

        metrics = {
            "val_accuracy": val_acc,
            "val_f1_macro": val_f1m,
            "model": "LogisticRegression(lbfgs)",
            "params": {
                "C": self.C,
                "max_iter": self.max_iter,
                "n_jobs": self.n_jobs,
            },
        }

        self.metrics_path.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8"
        )

        print(f"Saved model:   {self.model_path}")
        print(f"Saved metrics: {self.metrics_path}")
        print(f"VAL accuracy={val_acc:.4f}, f1_macro={val_f1m:.4f}")


def main(config_path: str):

    trainer = FashionTrainer(config_path)
    trainer.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.ini")
    args = parser.parse_args()
    main(args.config)