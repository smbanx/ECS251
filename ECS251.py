import torch
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
dataset = load_dataset("imdb")

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Select only the first 100 training and testing examples BEFORE tokenization
train_subset = dataset["train"].select(range(100))
test_subset = dataset["test"].select(range(100))

# Tokenization function
def preprocess_data(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# Tokenize only the first 100 training examples
tokenized_train = train_subset.map(preprocess_data, batched=True)

# Tokenize the first 100 test examples (for evaluation)
tokenized_test = test_subset.map(preprocess_data, batched=True)

# Convert to PyTorch format
tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_test = tokenized_test.rename_column("label", "labels")

tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Load pre-trained BERT with classification head
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Move model to GPU (if available)
model.to(device)

# Data collator (handles dynamic padding)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Compute metrics for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Training arguments with GPU acceleration
training_args = TrainingArguments(
    output_dir="./bert_sentiment",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,  # Adjust if running out of memory
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to="none",  # Disable reporting to Weights & Biases (optional)
    push_to_hub=False,  # Set True if you want to push to Hugging Face Hub
)

# Trainer with GPU support (Hugging Face handles CUDA automatically)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,  # Only 100 training samples
    eval_dataset=tokenized_test,  # First 100 test samples
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Save the model
model.save_pretrained("./bert_sentiment_model")
tokenizer.save_pretrained("./bert_sentiment_model")
