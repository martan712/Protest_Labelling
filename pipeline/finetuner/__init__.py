# gfx1150 (AMD Radeon 840M) has no ROCm kernels; present it as gfx1100. Set
# before torch is imported anywhere; no-op off ROCm.
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

from .data import load_data, build_dataloaders
from .model import get_device, load_tokenizer, train_model, evaluate_model, save_model
from .report import plot_loss, get_sample_predictions, generate_report_pdf
from .ollama import get_ollama_review

__all__ = [
    "load_data",
    "build_dataloaders",
    "get_device",
    "load_tokenizer",
    "train_model",
    "evaluate_model",
    "save_model",
    "plot_loss",
    "get_sample_predictions",
    "generate_report_pdf",
    "get_ollama_review",
]
