"""Loss plot, sample predictions and the PDF training report."""

import os
import random
import time

import torch
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, SimpleDocTemplate, PageBreak
from reportlab.lib.enums import TA_CENTER

from .data import TEXT_COLUMN


def plot_loss(train_losses, val_losses, test_loss, epochs, plot_path):
    plt.figure(figsize=(12, 7))
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, train_losses, label="Training Loss", marker="o", linestyle="-")
    plt.plot(epoch_range, val_losses, label="Validation Loss", marker="o", linestyle="--")
    plt.axhline(y=test_loss, color="r", linestyle=":", label=f"Final Test Loss ({test_loss:.4f})")
    plt.title("Training, Validation, and Test Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot saved to {plot_path}")
    return plot_path


def get_sample_predictions(test_df, model, tokenizer, device, class_names, num_samples=5):
    if len(test_df) == 0:
        print("Test set is empty, cannot get sample predictions.")
        return []

    sampled = test_df.iloc[random.sample(range(len(test_df)), min(num_samples, len(test_df)))]
    texts = sampled[TEXT_COLUMN].tolist()
    true_labels = sampled["label"].tolist()

    encodings = tokenizer(texts, truncation=True, max_length=512, padding="longest", return_tensors="pt")
    model.eval()
    with torch.no_grad():
        encodings = {k: v.to(device) for k, v in encodings.items()}
        predicted = model(**encodings).logits.argmax(dim=1).cpu().numpy()

    return [
        {"text": text, "true_label": class_names[true], "predicted_label": class_names[pred]}
        for text, true, pred in zip(texts, true_labels, predicted)
    ]


def _metrics_lines(field, metrics):
    """Indented bold-labelled lines for one metrics dict."""
    rows = [
        ("Average Loss", f"{metrics['avg_loss']:.4f}"),
        ("Accuracy", f"{metrics['accuracy']:.4f}"),
        ("Precision (weighted)", f"{metrics['precision']:.4f}"),
        ("Recall (weighted)", f"{metrics['recall']:.4f}"),
        ("F1-Score (weighted)", f"{metrics['f1']:.4f}"),
    ]
    return [field(label, value, indent=True) for label, value in rows]


def generate_report_pdf(pdf_path, plot_path, model_info, hyperparameters,
                        class_counts, val_metrics, test_metrics,
                        ollama_quality, ollama_description, sample_predictions):
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    normal.fontSize, normal.leading = 10, 12

    def heading(text):
        style = ParagraphStyle(name="h2", parent=styles["h2"], fontSize=18, leading=22,
                               spaceAfter=15, spaceBefore=25)
        return Paragraph(text, style)

    def field(label, value="", indent=False):
        pad = "&nbsp;" * 4 if indent else ""
        return Paragraph(f"{pad}<font name='Helvetica-Bold'>{label}:</font> {value}", normal)

    elements = []

    # Title page
    elements.append(Paragraph("Model Training Report", ParagraphStyle(
        name="h1", parent=styles["h1"], fontSize=24, leading=28, alignment=TA_CENTER, spaceAfter=20)))
    elements.append(Spacer(1, 0.25 * letter[1]))
    elements.append(Paragraph(f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", normal))
    elements.append(PageBreak())

    elements.append(heading("Model Details"))
    elements.append(field("Model Type", model_info["model_type"]))
    elements.append(field("Epochs Trained", model_info["epochs_trained"]))
    elements.append(field("Total Training Time", f"{model_info['training_time']:.2f} seconds"))
    elements.append(field("Device Used", model_info["device"]))

    elements.append(heading("Hyperparameters"))
    elements.append(field("Max Sequence Length", hyperparameters["max_len"]))
    elements.append(field("Batch Size", hyperparameters["batch_size"]))
    elements.append(field("Learning Rate", hyperparameters["learning_rate"]))

    elements.append(heading("Dataset Information"))
    split = (f"Train {hyperparameters['train_split']*100:.0f}% / "
             f"Validation {hyperparameters['val_split']*100:.0f}% / "
             f"Test {hyperparameters['test_split']*100:.0f}%")
    elements.append(field("Data Split", split))
    for split_name, counts in class_counts.items():
        elements.append(Spacer(1, 6))
        elements.append(field(f"Class Distribution ({split_name})"))
        for cls, count in counts.items():
            elements.append(field(str(cls), count, indent=True))
    elements.append(PageBreak())

    elements.append(heading("Performance Metrics"))
    if os.path.exists(plot_path):
        elements.append(Paragraph(f"<img src='{plot_path}' width='500' height='300'/>",
                                  ParagraphStyle(name="img", alignment=TA_CENTER)))
    else:
        elements.append(Paragraph("Loss plot image not found.", normal))
    elements.append(Spacer(1, 12))

    elements.append(field("Validation Set Metrics"))
    elements.extend(_metrics_lines(field, val_metrics))
    elements.append(Spacer(1, 10))
    elements.append(field("Test Set Metrics"))
    elements.extend(_metrics_lines(field, test_metrics))

    elements.append(heading("Sample Test Predictions"))
    if sample_predictions:
        for i, sample in enumerate(sample_predictions, 1):
            elements.append(field(f"Sample {i}"))
            elements.append(field("Text", sample["text"], indent=True))
            elements.append(field("True Category", sample["true_label"], indent=True))
            elements.append(field("Predicted Category", sample["predicted_label"], indent=True))
            elements.append(Spacer(1, 8))
    else:
        elements.append(Paragraph("No sample predictions available.", normal))

    elements.append(heading("Ollama Model Review"))
    elements.append(field("Overall Quality", ollama_quality))
    elements.append(field("Description"))
    elements.append(Paragraph(ollama_description, normal))

    doc.build(elements)
    print(f"Report PDF generated at {pdf_path}")
