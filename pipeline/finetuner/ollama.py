"""Optional qualitative model review via a local Ollama instance."""

import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:27b"


def _build_prompt(model_name, epochs, training_time, device, train_losses, val_losses,
                  test_loss, val_metrics, test_metrics, hp, class_names):
    def metrics_block(m):
        return (f"  Loss {m['avg_loss']:.4f}, Accuracy {m['accuracy']:.4f}, "
                f"Precision {m['precision']:.4f}, Recall {m['recall']:.4f}, F1 {m['f1']:.4f}")

    return f"""
You are an expert ML engineer reviewing a '{model_name}' text classification model
fine-tuned for sequence classification.

Training: {epochs} epochs, {training_time:.2f}s, on {device}.
Hyperparameters: max_len={hp['max_len']}, batch_size={hp['batch_size']}, lr={hp['learning_rate']},
split Train {hp['train_split']*100:.0f}% / Val {hp['val_split']*100:.0f}% / Test {hp['test_split']*100:.0f}%.

Train losses per epoch: {', '.join(f'{l:.4f}' for l in train_losses)}
Val losses per epoch:   {', '.join(f'{l:.4f}' for l in val_losses)}
Final test loss: {test_loss:.4f}

Validation metrics:
{metrics_block(val_metrics)}
Test metrics:
{metrics_block(test_metrics)}

The dataset has {len(class_names)} classes: {class_names}.

Classify overall quality as "Bad", "Mid" or "Perfect", then describe the model.
Consider overfitting (train loss << val/test), underfitting (high losses everywhere),
val/test consistency, and hyperparameter impact.

Respond as JSON: {{"quality": "Bad"|"Mid"|"Perfect", "description": "..."}}
"""


def get_ollama_review(model_name, epochs, training_time, device, train_losses, val_losses,
                      test_loss, val_metrics, test_metrics, hyperparameters, class_names):
    """Return (quality, description). Degrades gracefully if Ollama is unreachable."""
    prompt = _build_prompt(model_name, epochs, training_time, device, train_losses,
                           val_losses, test_loss, val_metrics, test_metrics,
                           hyperparameters, class_names)
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"}

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=420)
        response.raise_for_status()
        content = response.json().get("response", "{}")
        parsed = json.loads(content)
        return parsed.get("quality", "N/A"), parsed.get("description", "No description provided.")
    except json.JSONDecodeError:
        return "N/A", f"Ollama returned malformed JSON: {content}"
    except requests.exceptions.RequestException as e:
        return "N/A", f"Failed to reach Ollama: {e}"
    except Exception as e:
        return "N/A", f"Unexpected error during Ollama review: {e}"
