from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from clearml import Dataset, Task
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a sentiment model and log it to ClearML."
    )
    parser.add_argument("--project-name", type=str, default="itmo-mlops")
    parser.add_argument("--task-name", type=str, default="train-experiment")
    parser.add_argument("--dataset-id", type=str, required=True)
    parser.add_argument("--queue-name", type=str, default="students")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-remote", action="store_true")
    return parser.parse_args()


def load_dataset_frame(dataset_id: str) -> pd.DataFrame:
    dataset = Dataset.get(dataset_id=dataset_id)
    local_path = Path(dataset.get_local_copy())
    csv_files = sorted(local_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in dataset path: {local_path}")
    return pd.read_csv(csv_files[0])


def main() -> None:
    args = parse_args()

    task = Task.init(
        project_name=args.project_name,
        task_name=args.task_name,
        task_type=Task.TaskTypes.training,
    )

    params = task.connect(
        {
            "C": args.c,
            "max_iter": args.max_iter,
            "max_features": args.max_features,
            "dataset_id": args.dataset_id,
            "queue_name": args.queue_name,
        }
    )

    if task.running_locally() and not args.skip_remote:
        print(f"Sending task to queue: {args.queue_name}")
        task.execute_remotely(queue_name=args.queue_name, exit_process=True)

    df = load_dataset_frame(dataset_id=str(params["dataset_id"]))
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(max_features=int(params["max_features"]))
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(C=float(params["C"]), max_iter=int(params["max_iter"]))
    model.fit(x_train_vec, y_train)
    preds = model.predict(x_test_vec)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    logger = task.get_logger()
    logger.report_scalar("accuracy", "test", value=acc, iteration=0)
    logger.report_scalar("f1", "test", value=f1, iteration=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax)
    logger.report_matplotlib_figure("confusion_matrix", "test", figure=fig)
    plt.close(fig)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    model_path = outputs_dir / "model.pkl"
    with model_path.open("wb") as fp:
        pickle.dump({"model": model, "vectorizer": vectorizer}, fp)
    task.upload_artifact("model", artifact_object=str(model_path))

    print(f"acc={acc:.4f} f1={f1:.4f}")
    print(f"model artifact: {model_path}")
    task.close()


if __name__ == "__main__":
    main()
