from __future__ import annotations

import argparse

from clearml import OutputModel, Task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a model artifact to the ClearML Registry."
    )
    parser.add_argument(
        "--task-id",
        type=str,
        required=True,
        help="Training task ID holding the 'model' artifact.",
    )
    parser.add_argument("--model-name", type=str, default="sentiment-tfidf-logreg")
    parser.add_argument("--framework", type=str, default="scikit-learn")
    parser.add_argument(
        "--tags",
        type=str,
        default="sentiment,tfidf,logistic-regression",
        help="Comma-separated tags.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = Task.get_task(task_id=args.task_id)

    if "model" not in task.artifacts:
        raise KeyError(f"Task {args.task_id} has no 'model' artifact")

    artifact_path = task.artifacts["model"].get_local_copy()
    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]

    output_model = OutputModel(
        task=task,
        name=args.model_name,
        framework=args.framework,
        tags=tags,
    )
    output_model.update_weights(weights_filename=artifact_path)
    output_model.publish()

    print(f"Source task: {args.task_id}")
    print(f"Model ID: {output_model.id}")
    print("Published to Registry")


if __name__ == "__main__":
    main()
