import pandas as pd
import os
import time
import json
import random

# The integrated AMD Radeon 840M is gfx1150 (RDNA 3.5), which the ROCm build of
# PyTorch has no precompiled kernels for. Presenting it as gfx1100 lets it run on
# the gfx1100 kernels. Must be set before torch is imported. No-op on non-ROCm.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, SimpleDocTemplate, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import requests

"""
This script is designed to train a Hugging Face Transformer model for text classification tasks.
It includes data loading, preprocessing, model training, evaluation, and generating a PDF report with performance metrics and visualizations.
It also integrates with Ollama to get a model review based on the training results.

To understand the code:
1. It starts by defining global mappings for class indices to names and vice versa.
2. It defines a custom dataset class for handling the data.
3. It includes functions for loading data, tokenizing text, creating data loaders, training the model, evaluating it, and saving the model.
4. It generates a loss plot and sample predictions.
5. It integrates with Ollama to get a model review based on the training results.
6. Finally, it generates a PDF report summarizing the training process, metrics, and Ollama review.

The script is structured to be modular, allowing for easy adjustments to hyperparameters, model types, and data sources.
It also includes error handling for robustness, especially when loading models and tokenizers.
The script is intended to be run as a standalone program, and it expects a CSV file with text data and class labels for training.
The CSV file should have columns named 'class' and 'clean_notes', where 'class' contains the class labels and 'clean_notes' contains the text data.
The script will create a directory for saving the model and report, and it will generate a PDF report summarizing the training results, including a loss plot and sample predictions.
The Ollama integration provides an additional layer of analysis, generating a qualitative review of the model based on its performance metrics.
"""

GLOBAL_INDEX_TO_CLASS = {
    1: 'environment',
    2: 'Culture',
    3: 'Education',
    4: 'Palestine-Israel Conflict',
    5: 'Labor Rights',
    6: 'Public Services & Social Welfare',
    7: 'Justice & Civil Rights',
    8: 'Climate Action & Animal Welfare',
    9: 'Political & Democratic Governance',
    10: 'Ukraine-Russia War',
    11: 'Infrastructure',
    12: 'Climate Action & Resource Management',
}
GLOBAL_CLASS_TO_INDEX = {v: k for k, v in GLOBAL_INDEX_TO_CLASS.items()}


# --- Dataset and DataLoader Classes ---
class AGNewsDatasetHF(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# --- Data Loading and Preprocessing ---
def load_data(file_path, train_split=0.8, val_split=0.1, random_state=42):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        df = df[['class', 'clean_notes']]
        df = df[df['class'] != 'NoN']
        print("Test")
        unique_classes = sorted(df['class'].unique())
        class_names = sorted(df['class'].map(lambda x: df[df['class'] == x]['class'].iloc[0]).unique())
        
        print("Test2")
        if 'class' in df.columns:
            class_names = sorted(df['class'].unique())
            GLOBAL_INDEX_TO_CLASS.clear()
            GLOBAL_INDEX_TO_CLASS.update({i + 1: name for i, name in enumerate(class_names)})
            GLOBAL_CLASS_TO_INDEX.clear()
            GLOBAL_CLASS_TO_INDEX.update({name: i + 1 for i, name in enumerate(class_names)})

        print(f"Unique classes found: {unique_classes}")
        df['class'] = df['class'].map(GLOBAL_CLASS_TO_INDEX).astype(int)
        for index, class_name in GLOBAL_INDEX_TO_CLASS.items():
           print(f"Class {index}: {class_name}")

        
        total_size = len(df)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size

        train_df = df.sample(n=train_size, random_state=random_state)
        remaining_df = df.drop(train_df.index)
        val_df = remaining_df.sample(n=val_size, random_state=random_state)
        test_df = remaining_df.drop(val_df.index)

        print(f"Dataset loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        train_class_counts = train_df['class'].map(GLOBAL_INDEX_TO_CLASS).value_counts().to_dict()
        val_class_counts = val_df['class'].map(GLOBAL_INDEX_TO_CLASS).value_counts().to_dict()
        test_class_counts = test_df['class'].map(GLOBAL_INDEX_TO_CLASS).value_counts().to_dict()

        return train_df, val_df, test_df, \
               train_class_counts, val_class_counts, test_class_counts
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")

def create_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception as e:
        raise ValueError(f"Could not load tokenizer for model '{model_name}'. Please check the model name. Error: {e}")


def tokenize_function(texts, tokenizer, max_len):
    return tokenizer(texts,
                     padding='max_length',
                     truncation=True,
                     max_length=max_len,
                     return_tensors='pt')

def tokenize_dataset(train_df, val_df, test_df, tokenizer, max_len):
    train_encodings = tokenize_function(train_df['clean_notes'].tolist(), tokenizer, max_len)
    val_encodings = tokenize_function(val_df['clean_notes'].tolist(), tokenizer, max_len)
    test_encodings = tokenize_function(test_df['clean_notes'].tolist(), tokenizer, max_len)
    y_train_hf = torch.tensor((train_df['class'] - 1).values, dtype=torch.long)
    y_val_hf = torch.tensor((val_df['class'] - 1).values, dtype=torch.long)
    y_test_hf = torch.tensor((test_df['class'] - 1).values, dtype=torch.long)
    return train_encodings, y_train_hf, val_encodings, y_val_hf, test_encodings, y_test_hf

def create_data_loaders(train_encodings, y_train_hf, val_encodings, y_val_hf, test_encodings, y_test_hf, batch_size=16):
    train_dataset_hf = AGNewsDatasetHF(train_encodings, y_train_hf)
    val_dataset_hf = AGNewsDatasetHF(val_encodings, y_val_hf)
    test_dataset_hf = AGNewsDatasetHF(test_encodings, y_test_hf)

    train_loader_hf = DataLoader(train_dataset_hf, batch_size=batch_size, shuffle=True)
    val_loader_hf = DataLoader(val_dataset_hf, batch_size=batch_size, shuffle=False)
    test_loader_hf = DataLoader(test_dataset_hf, batch_size=batch_size, shuffle=False)

    return train_loader_hf, val_loader_hf, test_loader_hf

# --- Model Training and Evaluation ---
def train_model(tokenizer_name, num_labels, train_loader, val_loader, learning_rate, epochs, output_dir=None, tokenizer=None):
    model_name = tokenizer_name
    try:
        model_hf = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    except Exception as e:
        raise ValueError(f"Could not load model for '{model_name}'. Please check the model name or ensure it's suitable for sequence classification. Error: {e}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "MPS (Apple Silicon GPU)"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    model_hf.to(device)
    optimizer_hf = AdamW(model_hf.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    start_time = time.time()

    print("\nStarting training for Hugging Face Transformer Model...")
    for epoch in range(epochs):
        model_hf.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer_hf.zero_grad(set_to_none=True)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # bf16 autocast roughly halves GPU compute/memory with no change to the
            # training procedure; bf16 needs no GradScaler. No-op on CPU.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                outputs = model_hf(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer_hf.step()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}: Average Training Loss: {avg_train_loss:.4f}")

        avg_val_loss, _, _, _, _ = evaluate_model(model_hf, val_loader, device, return_metrics=True)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}: Average Validation Loss: {avg_val_loss:.4f}")

        # Checkpoint after every epoch so an interrupt/crash doesn't lose progress.
        # Overwrites the same location each time, so it doubles as the final model.
        if output_dir is not None and tokenizer is not None:
            save_model(model_hf, "hf_transformer_model", tokenizer, output_dir)
            print(f"Epoch {epoch+1}: checkpoint saved to {os.path.join(output_dir, 'hf_transformer_model')}")

    end_time = time.time()
    training_time = end_time - start_time

    print("Hugging Face Transformer Training finished.")
    return model_hf, train_losses, val_losses, device_name, training_time

def evaluate_model(model, data_loader, device, return_metrics=False):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            _, predicted = torch.max(logits, 1)

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predictions)
    labels_in_data = list(set(true_labels) | set(predictions))
    if not labels_in_data:
        precision, recall, f1 = 0.0, 0.0, 0.0
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0,
            labels=sorted(list(set(true_labels) | set(predictions)))
        )

    if return_metrics:
        return avg_loss, accuracy, precision, recall, f1
    else:
        return avg_loss, accuracy, precision, recall, f1

def save_model(model, name, tokenizer, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_output_dir = os.path.join(output_dir, name)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    print(f"Saving model to {model_output_dir}...")
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print("Model and tokenizer saved successfully!")

# --- Plotting ---
def plot_loss(train_losses, val_losses, test_loss, epochs, plot_path="loss_plot.png"):
    plt.figure(figsize=(12, 7))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', marker='o', linestyle='-')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='o', linestyle='--')
    
    if isinstance(test_loss, (int, float)):
        plt.axhline(y=test_loss, color='r', linestyle=':', label=f'Final Test Loss ({test_loss:.4f})')
    elif isinstance(test_loss, list) and len(test_loss) == epochs: # If test loss was tracked per epoch
        plt.plot(range(1, epochs + 1), test_loss, label='Test Loss', marker='x', linestyle='-.', color='purple')
    
    plt.title('Training, Validation, and Test Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot saved to {plot_path}")
    return plot_path

# --- Sample Prediction Function ---
def get_sample_predictions(test_df, model, tokenizer, device, num_samples=5):
    samples = []
    if len(test_df) == 0:
        print("Test DataFrame is empty, cannot get sample predictions.")
        return samples

    sample_indices = random.sample(range(len(test_df)), min(num_samples, len(test_df)))
    sampled_df = test_df.iloc[sample_indices]

    texts = sampled_df['clean_notes'].tolist()
    true_labels_indices = (sampled_df['class'] - 1).tolist()

    encodings = tokenize_function(texts, tokenizer, max_len=tokenizer.model_max_length)
    
    model.eval()
    with torch.no_grad():
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predicted_indices = torch.max(logits, 1)
        predicted_indices = predicted_indices.cpu().numpy()

    for i in range(len(texts)):
        true_label_name = GLOBAL_INDEX_TO_CLASS.get(true_labels_indices[i] + 1, "Unknown")
        predicted_label_name = GLOBAL_INDEX_TO_CLASS.get(predicted_indices[i] + 1, "Unknown")
        samples.append({
            'text': texts[i],
            'true_label': true_label_name,
            'predicted_label': predicted_label_name
        })
    return samples


# --- Ollama Integration ---
def get_ollama_review(model_name, epochs, training_time, device,
                      train_losses, val_losses, test_loss,
                      val_metrics, test_metrics, hyperparameters,
                      num_classes, class_names):
    
    ollama_url = "http://localhost:11434/api/generate"
    model_review = "Could not get review from Ollama."
    model_quality = "N/A"

    prompt = f"""
    You are an expert Machine Learning Engineer reviewing a text classification model.
    The model is a '{model_name}' fine-tuned for sequence classification.

    Here are the training details and performance metrics:
    - Model Type: {model_name}
    - Epochs Trained: {epochs}
    - Total Training Time: {training_time:.2f} seconds
    - Device Used: {device}

    --- Hyperparameters ---
    - Max Sequence Length: {hyperparameters['max_len']}
    - Batch Size: {hyperparameters['batch_size']}
    - Learning Rate: {hyperparameters['learning_rate']}
    - Data Split: Train {hyperparameters['train_split']*100:.0f}% / Validation {hyperparameters['val_split']*100:.0f}% / Test {hyperparameters['test_split']*100:.0f}%

    --- Loss Values ---
    - Average Training Losses per Epoch: {', '.join(f'{l:.4f}' for l in train_losses)}
    - Average Validation Losses per Epoch: {', '.join(f'{l:.4f}' for l in val_losses)}
    - Final Test Loss: {test_loss:.4f}

    --- Validation Metrics ---
    - Validation Average Loss: {val_metrics['avg_loss']:.4f}
    - Validation Accuracy: {val_metrics['accuracy']:.4f}
    - Validation Precision (weighted): {val_metrics['precision']:.4f}
    - Validation Recall (weighted): {val_metrics['recall']:.4f}
    - Validation F1-Score (weighted): {val_metrics['f1']:.4f}

    --- Test Metrics ---
    - Test Average Loss: {test_metrics['avg_loss']:.4f}
    - Test Accuracy: {test_metrics['accuracy']:.4f}
    - Test Precision (weighted): {test_metrics['precision']:.4f}
    - Test Recall (weighted): {test_metrics['recall']:.4f}
    - Test F1-Score (weighted): {test_metrics['f1']:.4f}

    The dataset has {num_classes} classes: {class_names}.

    Based on these metrics, provide a concise review of the model's performance.
    
    First, classify the model's overall quality as one of these: "Bad", "Mid", "Perfect".
    Then, provide a descriptive analysis.
    Consider if the model is overfitting (training loss significantly lower than validation/test), underfitting (high losses across the board), or performing well.
    Comment on the consistency between validation and test metrics.
    Mention any concerns or strengths regarding the hyperparameters and their impact on performance.

    Format your output as a JSON object with two keys:
    "quality": "Bad" | "Mid" | "Perfect"
    "description": "Your detailed description here."
    """

    data = {
        "model": "gemma3:27b",
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(ollama_url, headers=headers, data=json.dumps(data), timeout=420)
        response.raise_for_status()
        result = response.json()
        
        generated_content = result.get('response', '{}')
        try:
            parsed_response = json.loads(generated_content)
            model_quality = parsed_response.get('quality', 'N/A')
            model_review = parsed_response.get('description', 'No description provided.')
        except json.JSONDecodeError:
            model_review = f"Ollama returned non-JSON or malformed JSON: {generated_content}"
            print(f"DEBUG: Ollama response was not pure JSON: {generated_content}")

    except requests.exceptions.RequestException as e:
        model_review = f"Failed to connect to Ollama or request timed out: {e}"
        print(model_review)
    except Exception as e:
        model_review = f"An unexpected error occurred during Ollama review: {e}"
        print(model_review)

    return model_quality, model_review

# --- PDF Report Generation ---
def generate_report_pdf(pdf_path, plot_path, model_info, hyperparameters,
                        train_class_counts, val_class_counts, test_class_counts,
                        val_metrics, test_metrics,
                        ollama_quality, ollama_description,
                        sample_predictions):

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()

    h1_style = ParagraphStyle(
        name='h1',
        parent=styles['h1'],
        fontSize=24,
        leading=28,
        alignment=TA_CENTER,
        spaceAfter=20
    )
    h2_style = ParagraphStyle(
        name='h2',
        parent=styles['h2'],
        fontSize=18,
        leading=22,
        spaceAfter=15,
        spaceBefore=25
    )
    h3_style = ParagraphStyle(
        name='h3',
        parent=styles['h3'],
        fontSize=14,
        leading=16,
        spaceAfter=10,
        spaceBefore=15
    )
    normal_bold = ParagraphStyle(
        name='NormalBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12
    )
    normal_text = styles['Normal']
    normal_text.fontSize = 10
    normal_text.leading = 12

    elements = []

    # Title Page
    elements.append(Paragraph("Model Training Report", h1_style))
    elements.append(Spacer(1, 0.5 * letter[1] / 2)) # Vertical spacer to center
    elements.append(Paragraph(f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(PageBreak())

    # --- Section: Model Details ---
    elements.append(Paragraph("Model Details", h2_style))
    elements.append(Paragraph(f"<font name='Helvetica-Bold'>Model Type:</font> {model_info['model_type']}", normal_text))
    elements.append(Paragraph(f"<font name='Helvetica-Bold'>Epochs Trained:</font> {model_info['epochs_trained']}", normal_text))
    elements.append(Paragraph(f"<font name='Helvetica-Bold'>Total Training Time:</font> {model_info['training_time']:.2f} seconds", normal_text))
    elements.append(Paragraph(f"<font name='Helvetica-Bold'>Device Used:</font> {model_info['device']}", normal_text))
    elements.append(Spacer(1, 12))

    # --- Section: Hyperparameters ---
    elements.append(Paragraph("Hyperparameters", h2_style))
    elements.append(Paragraph(f"<font name='Helvetica-Bold'>Max Sequence Length:</font> {hyperparameters['max_len']}", normal_text))
    elements.append(Paragraph(f"<font name='Helvetica-Bold'>Batch Size:</font> {hyperparameters['batch_size']}", normal_text))
    elements.append(Paragraph(f"<font name='Helvetica-Bold'>Learning Rate:</font> {hyperparameters['learning_rate']}", normal_text))
    elements.append(Spacer(1, 12))

    # --- Section: Dataset Information ---
    elements.append(Paragraph("Dataset Information", h2_style))
    elements.append(Paragraph(f"<font name='Helvetica-Bold'>Data Split:</font> Train {hyperparameters['train_split']*100:.0f}% / Validation {hyperparameters['val_split']*100:.0f}% / Test {hyperparameters['test_split']*100:.0f}%", normal_text))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<font name='Helvetica-Bold'>Class Distribution (Training Set):</font>", normal_text))
    for cls, count in train_class_counts.items():
        elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{cls}: {count}", normal_text))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("<font name='Helvetica-Bold'>Class Distribution (Validation Set):</font>", normal_text))
    for cls, count in val_class_counts.items():
        elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{cls}: {count}", normal_text))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("<font name='Helvetica-Bold'>Class Distribution (Test Set):</font>", normal_text))
    for cls, count in test_class_counts.items():
        elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{cls}: {count}", normal_text))
    elements.append(Spacer(1, 12))
    elements.append(PageBreak())
    elements.append(Paragraph("Performance Metrics", h2_style))
    
    elements.append(Paragraph("<font name='Helvetica-Bold'>Graph:</font>", normal_text))
    # --- Section: Loss Plot ---
    if os.path.exists(plot_path):
        elements.append(Spacer(1, 280))

        img = ImageReader(plot_path)
        img_width, img_height = img.getSize()
        
        max_plot_width = 500
        max_plot_height = 300
        aspect_ratio = img_height / img_width

        plot_draw_width = img_width
        plot_draw_height = img_height

        if img_width > max_plot_width:
            plot_draw_width = max_plot_width
            plot_draw_height = plot_draw_width * aspect_ratio
        
        if plot_draw_height > max_plot_height:
            plot_draw_height = max_plot_height
            plot_draw_width = plot_draw_height / aspect_ratio

        elements.append(
            Paragraph(f"<img src='{plot_path}' width='{plot_draw_width}' height='{plot_draw_height}'/>",
            ParagraphStyle(name='img_style', alignment=TA_CENTER))
        )
        elements.append(Spacer(1, 12))
    else:
        elements.append(Paragraph("Loss plot image not found.", normal_text))
    # --- Section: Performance Metrics ---
    
    elements.append(Paragraph("<font name='Helvetica-Bold'>Validation Set Metrics:</font>", normal_text))
    elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<font name='Helvetica-Bold'>Average Loss:</font> {val_metrics['avg_loss']:.4f}", normal_text))
    elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<font name='Helvetica-Bold'>Accuracy:</font> {val_metrics['accuracy']:.4f}", normal_text))
    elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<font name='Helvetica-Bold'>Precision (weighted):</font> {val_metrics['precision']:.4f}", normal_text))
    elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<font name='Helvetica-Bold'>Recall (weighted):</font> {val_metrics['recall']:.4f}", normal_text))
    elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<font name='Helvetica-Bold'>F1-Score (weighted):</font> {val_metrics['f1']:.4f}", normal_text))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("<font name='Helvetica-Bold'>Test Set Metrics:</font>", normal_text))
    elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<font name='Helvetica-Bold'>Average Loss:</font> {test_metrics['avg_loss']:.4f}", normal_text))
    elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<font name='Helvetica-Bold'>Accuracy:</font> {test_metrics['accuracy']:.4f}", normal_text))
    elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<font name='Helvetica-Bold'>Precision (weighted):</font> {test_metrics['precision']:.4f}", normal_text))
    elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<font name='Helvetica-Bold'>Recall (weighted):</font> {test_metrics['recall']:.4f}", normal_text))
    elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;<font name='Helvetica-Bold'>F1-Score (weighted):</font> {test_metrics['f1']:.4f}", normal_text))
    elements.append(Spacer(1, 12))
    

    # --- Section: Sample Predictions ---
    elements.append(Paragraph("Sample Test Predictions", h2_style))
    if sample_predictions:
        for i, sample in enumerate(sample_predictions):
            elements.append(Paragraph(f"<font name='Helvetica-Bold'>Sample {i+1}:</font>", normal_text))
            elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;Text: {sample['text']}", normal_text))
            elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;True Category: <font name='Helvetica-Bold'>{sample['true_label']}</font>", normal_text))
            elements.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;Predicted Category: <font name='Helvetica-Bold'>{sample['predicted_label']}</font>", normal_text))
            elements.append(Spacer(1, 8))
    else:
        elements.append(Paragraph("No sample predictions available (e.g., empty test set or error during retrieval).", normal_text))
    elements.append(Spacer(1, 12))

    # --- Section: Ollama Model Review ---
    elements.append(Paragraph("Ollama Model Review", h2_style))
    elements.append(Paragraph(f"<font name='Helvetica-Bold'>Overall Quality:</font> {ollama_quality}", normal_text))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph("<font name='Helvetica-Bold'>Description:</font>", normal_text))
    elements.append(Paragraph(ollama_description, normal_text)) # Ollama description is now simply a paragraph
    
    doc.build(elements)
    print(f"Report PDF generated at {pdf_path}")

# --- Main Execution ---
def main():
    file_path = '../data/labeled_balanced.csv'
    max_len = 128
    batch_size = 16
    learning_rate = 5e-5
    
    train_split_perc = 0.7
    val_split_perc = 0.2
    test_split_perc = 0.1

    default_name = 'hf_transformer_enhanced'
    loss_plot_name = 'training_validation_test_loss.png'

    # --- User Input ---
    tokenizer_name = input("Enter the Hugging Face model name (e.g., 'bert-base-uncased', 'roberta-base'): ").strip()
    while not tokenizer_name:
        tokenizer_name = input("Model name cannot be empty. Please enter a valid Hugging Face model name: ").strip()
    
    name = input(f"Enter the model name for saving the model and report (default: {default_name}): ").strip() or default_name
    output_dir = f"models/{name}"
    report_pdf_name = f"{name}_report.pdf"
    os.makedirs(output_dir, exist_ok=True)


    epochs_input = input("Enter the number of training epochs (e.g., 3): ").strip()
    epochs = 0
    while True:
        try:
            epochs = int(epochs_input)
            if epochs <= 0:
                print("Number of epochs must be a positive integer.")
                epochs_input = input("Enter the number of training epochs (e.g., 3): ").strip()
            else:
                break
        except ValueError:
            print("Invalid input. Please enter an integer for epochs.")
            epochs_input = input("Enter the number of training epochs (e.g., 3): ").strip()

    # 1. Load Data with Validation Split and Class Counts
    train_df, val_df, test_df, \
    train_class_counts, val_class_counts, test_class_counts = \
        load_data(file_path, train_split=train_split_perc, val_split=val_split_perc)
    
    tokenizer = create_tokenizer(tokenizer_name)
    
    # 2. Tokenize Datasets
    train_encodings, y_train_hf, val_encodings, y_val_hf, test_encodings, y_test_hf = \
        tokenize_dataset(train_df, val_df, test_df, tokenizer, max_len)
    
    # 3. Create Data Loaders
    train_loader_hf, val_loader_hf, test_loader_hf = \
        create_data_loaders(train_encodings, y_train_hf, val_encodings, y_val_hf, test_encodings, y_test_hf, batch_size)

    num_labels = len(GLOBAL_INDEX_TO_CLASS)
    
    # 4. Train Model with Validation Tracking
    model_hf, train_losses, val_losses, device_name, training_time = train_model(
        tokenizer_name, num_labels, train_loader_hf, val_loader_hf, learning_rate, epochs,
        output_dir=output_dir, tokenizer=tokenizer
    )
    
    # 5. Save Model
    save_model(model_hf, "hf_transformer_model", tokenizer, output_dir)
    
    # Determine the device for evaluation
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 6. Evaluate Model on Test Set (final metrics)
    test_avg_loss, test_accuracy, test_precision, test_recall, test_f1 = \
        evaluate_model(model_hf, test_loader_hf, device, return_metrics=True)
    print(f"Final Test Metrics: Loss={test_avg_loss:.4f}, Acc={test_accuracy:.4f}, P={test_precision:.4f}, R={test_recall:.4f}, F1={test_f1:.4f}")

    # 7. Evaluate Model on Validation Set (final metrics for report)
    val_avg_loss, val_accuracy, val_precision, val_recall, val_f1 = \
        evaluate_model(model_hf, val_loader_hf, device, return_metrics=True)
    print(f"Final Validation Metrics: Loss={val_avg_loss:.4f}, Acc={val_accuracy:.4f}, P={val_precision:.4f}, R={val_recall:.4f}, F1={val_f1:.4f}")

    # 8. Generate and save the loss plot
    plot_path = os.path.join(output_dir, loss_plot_name)
    plot_loss(train_losses, val_losses, test_avg_loss, epochs, plot_path)

    # 9. Prepare info for PDF report
    model_info = {
        'model_type': tokenizer_name,
        'epochs_trained': epochs,
        'training_time': training_time,
        'device': device_name
    }
    hyperparameters = {
        'max_len': max_len,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_split': train_split_perc,
        'val_split': val_split_perc,
        'test_split': test_split_perc
    }
    val_metrics = {
        'avg_loss': val_avg_loss, 'accuracy': val_accuracy,
        'precision': val_precision, 'recall': val_recall, 'f1': val_f1
    }
    test_metrics = {
        'avg_loss': test_avg_loss, 'accuracy': test_accuracy,
        'precision': test_precision, 'recall': test_recall, 'f1': test_f1
    }
    
    # 10. Get Sample Predictions for PDF
    num_samples_to_show = 5
    sample_predictions = get_sample_predictions(test_df, model_hf, tokenizer, device, num_samples_to_show)
    
    # 11. Get Ollama Model Review
    print("\nAttempting to get model review from Ollama (this might take a moment)...")
    class_names_str = ", ".join([f"'{GLOBAL_INDEX_TO_CLASS[i]}'" for i in sorted(GLOBAL_INDEX_TO_CLASS.keys())])
    ollama_quality, ollama_description = get_ollama_review(
        tokenizer_name, epochs, training_time, device_name,
        train_losses, val_losses, test_avg_loss,
        val_metrics, test_metrics, hyperparameters,
        len(GLOBAL_INDEX_TO_CLASS), class_names_str
    )
    print(f"Ollama Quality: {ollama_quality}")
    print(f"Ollama Description Preview: {ollama_description[:200]}...")


    # 12. Generate the PDF report
    pdf_output_path = os.path.join(output_dir, report_pdf_name)
    generate_report_pdf(
        pdf_output_path, plot_path, model_info, hyperparameters,
        train_class_counts, val_class_counts, test_class_counts,
        val_metrics, test_metrics,
        ollama_quality, ollama_description,
        sample_predictions
    )

if __name__ == '__main__':
    main()