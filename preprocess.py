"""Stage 4: ClearML Serving custom-engine preprocessing for the sentiment model.

The published artifact is a pickled bundle ``{"model": ..., "vectorizer": ...}``,
so a custom engine is required: TF-IDF vectorization and probability decoding are
handled here instead of relying on the built-in sklearn engine.
"""

from __future__ import annotations

import pickle
from typing import Any, Callable, Optional


class Preprocess:
    """Run before and after every request on the ClearML inference service.

    Execution flow is synchronous:
    ``preprocess(body) -> process(data) -> postprocess(data) -> result``.
    """

    def __init__(self) -> None:
        """Initialise per-service state (loaded once, shared across requests)."""
        self._model = None
        self._vectorizer = None

    def load(self, local_file_name: str) -> None:
        """Load the pickled ``{model, vectorizer}`` bundle from the registry weights."""
        with open(local_file_name, "rb") as fp:
            bundle = pickle.load(fp)
        self._model = bundle["model"]
        self._vectorizer = bundle["vectorizer"]

    def preprocess(
        self,
        body: dict,
        state: dict,
        collect_custom_statistics_fn: Optional[Callable[[dict], None]] = None,
    ) -> str:
        """Extract the raw text from the request body."""
        text = body.get("text", "")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Request body must contain a non-empty 'text' field")
        return text

    def process(
        self,
        data: str,
        state: dict,
        collect_custom_statistics_fn: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """Vectorize the text and run the classifier, returning label and scores."""
        features = self._vectorizer.transform([data])
        probabilities = self._model.predict_proba(features)[0]
        classes = list(self._model.classes_)
        best_index = int(probabilities.argmax())
        return {
            "label": str(classes[best_index]),
            "confidence": float(probabilities[best_index]),
            "probabilities": {
                str(label): float(score) for label, score in zip(classes, probabilities)
            },
        }

    def postprocess(
        self,
        data: dict,
        state: dict,
        collect_custom_statistics_fn: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """Return the prediction dictionary as the REST API response."""
        return data
