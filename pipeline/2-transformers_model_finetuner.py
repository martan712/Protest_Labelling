"""Fine-tune a Hugging Face transformer for protest-note classification.

Loads the labeled CSV, trains a sequence-classification model, evaluates it, and
writes a PDF report (loss plot, metrics, sample predictions, optional Ollama
review). The saved model stores real class names in its config, so the labeling
notebook (3-labeling_by_transformer_model.ipynb) needs no CSV reconstruction.

Run from the pipeline/ directory:  python 2-transformers_model_finetuner.py
"""

import os

from finetuner import (
    load_data, build_dataloaders, get_device, load_tokenizer, train_model,
    evaluate_model, save_model, plot_loss, get_sample_predictions,
    generate_report_pdf, get_ollama_review,
)

DATA_FILE = "../data/labeled_balanced_20.csv"  # balanced 20-class set (from dataset_balancing.ipynb)
MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.7, 0.2, 0.1
LOSS_PLOT_NAME = "training_validation_test_loss.png"
DEFAULT_NAME = "hf_transformer_enhanced"


def prompt_nonempty(message):
    value = input(message).strip()
    while not value:
        value = input("Value cannot be empty. " + message).strip()
    return value


def prompt_positive_int(message):
    while True:
        try:
            value = int(input(message).strip())
            if value > 0:
                return value
            print("Must be a positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")


def main():
    model_name = prompt_nonempty("Hugging Face model name (e.g. 'bert-base-uncased', 'roberta-base'): ")
    name = input(f"Name for saving model and report (default: {DEFAULT_NAME}): ").strip() or DEFAULT_NAME
    epochs = prompt_positive_int("Number of training epochs (e.g. 3): ")

    output_dir = f"models/{name}"
    os.makedirs(output_dir, exist_ok=True)

    splits = load_data(DATA_FILE, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT)
    device, device_name = get_device()
    print(f"Using device: {device_name}")

    tokenizer = load_tokenizer(model_name)
    train_loader, val_loader, test_loader = build_dataloaders(splits, tokenizer, MAX_LEN, BATCH_SIZE)

    model, train_losses, val_losses, training_time = train_model(
        model_name, splits.class_names, train_loader, val_loader, device,
        LEARNING_RATE, epochs, output_dir=output_dir, tokenizer=tokenizer,
    )
    save_model(model, tokenizer, output_dir)

    test_metrics = evaluate_model(model, test_loader, device)
    val_metrics = evaluate_model(model, val_loader, device)
    print(f"Test metrics:  {test_metrics}")
    print(f"Val metrics:   {val_metrics}")

    plot_path = plot_loss(train_losses, val_losses, test_metrics["avg_loss"], epochs,
                          os.path.join(output_dir, LOSS_PLOT_NAME))

    hyperparameters = {
        "max_len": MAX_LEN, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "train_split": TRAIN_SPLIT, "val_split": VAL_SPLIT, "test_split": TEST_SPLIT,
    }
    model_info = {
        "model_type": model_name, "epochs_trained": epochs,
        "training_time": training_time, "device": device_name,
    }
    class_counts = {
        "Training Set": splits.class_counts(splits.train),
        "Validation Set": splits.class_counts(splits.val),
        "Test Set": splits.class_counts(splits.test),
    }

    samples = get_sample_predictions(splits.test, model, tokenizer, device, splits.class_names)

    print("\nRequesting Ollama review (this may take a moment)...")
    ollama_quality, ollama_description = get_ollama_review(
        model_name, epochs, training_time, device_name, train_losses, val_losses,
        test_metrics["avg_loss"], val_metrics, test_metrics, hyperparameters, splits.class_names,
    )
    print(f"Ollama quality: {ollama_quality}")

    generate_report_pdf(
        os.path.join(output_dir, f"{name}_report.pdf"), plot_path, model_info, hyperparameters,
        class_counts, val_metrics, test_metrics, ollama_quality, ollama_description, samples,
    )


if __name__ == "__main__":
    main()
