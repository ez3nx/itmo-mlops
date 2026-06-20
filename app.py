from __future__ import annotations

import os
import time

import gradio as gr
import requests

ENDPOINT = os.getenv("INFERENCE_ENDPOINT", "http://127.0.0.1:9090/serve/sentiment")
REQUEST_TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "10"))


def predict(text: str) -> tuple[dict, str]:
    """Send text to the inference endpoint and return label scores with latency."""
    if not text or not text.strip():
        return {}, "Enter some text first."

    try:
        start = time.perf_counter()
        response = requests.post(ENDPOINT, json={"text": text}, timeout=REQUEST_TIMEOUT)
        latency_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return {}, f"Endpoint unavailable at {ENDPOINT}. Is the serving stack running?"
    except requests.exceptions.Timeout:
        return {}, f"Endpoint timed out after {REQUEST_TIMEOUT:.0f}s."
    except requests.exceptions.RequestException as exc:
        return {}, f"Request failed: {exc}"

    payload = response.json()
    probabilities = payload.get("probabilities") or {
        payload.get("label", "label"): payload.get("confidence", 0.0)
    }
    status = f"Predicted: {payload.get('label', '?')} | latency: {latency_ms:.1f} ms"
    return probabilities, status


def build_ui() -> gr.Blocks:
    """Assemble the Gradio interface."""
    with gr.Blocks(title="Sentiment Classifier") as demo:
        gr.Markdown(
            "# Sentiment Classifier\n"
            "Three-class tweet sentiment served from the ClearML Model Registry "
            "via ClearML Serving. The UI talks to the endpoint over HTTP only."
        )
        text_input = gr.Textbox(label="Text", lines=4, placeholder="Type a sentence...")
        predict_button = gr.Button("Predict", variant="primary")
        label_output = gr.Label(label="Class probabilities", num_top_classes=3)
        status_output = gr.Markdown()

        predict_button.click(
            fn=predict, inputs=text_input, outputs=[label_output, status_output]
        )
        text_input.submit(
            fn=predict, inputs=text_input, outputs=[label_output, status_output]
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
