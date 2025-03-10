import os
import mmap
import time
import torch
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set visible GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Argument parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--profile", action="store_true", help="Enable PyTorch Profiler")
args = parser.parse_args()

# Optimize thread synchronization to reduce `futex` syscalls
torch.set_num_threads(4)

# Optimize GPU execution
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Preallocate memory to reduce `brk()` and `munmap()` calls
preallocated_tensor = torch.empty((10000, 512), dtype=torch.float16, device="cuda")

# Cache timestamp to reduce `clock_gettime()` calls
cached_time = time.time()
def get_cached_time():
    return cached_time

# Function to use `mmap()` for fast dataset file reads
# def memory_map_file(filename):
#     with open(filename, "r+b") as f:
#         return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
# 
# # Set dataset cache path
# dataset_path = "./cache/dataset_file.txt"
# if os.path.exists(dataset_path):
#     mapped_dataset = memory_map_file(dataset_path)

# Load dataset
dataset = load_dataset("imdb")

# Load tokenizer with caching
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="./cache")

# Select only the first 100 training and testing examples BEFORE tokenization
train_subset = dataset["train"].select(range(100))
test_subset = dataset["test"].select(range(100))

# Tokenization function
def preprocess_data(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# Tokenize datasets using `mmap()` if available
tokenized_train = train_subset.map(preprocess_data, batched=True)
tokenized_test = test_subset.map(preprocess_data, batched=True)

# Convert to PyTorch format
tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_test = tokenized_test.rename_column("label", "labels")

tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Load pre-trained BERT model with classification head
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to("cuda")

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
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=1000,  # Reduced logging frequency to reduce `write()` overhead
    load_best_model_at_end=True,
    save_total_limit=1,  # Reduce checkpoint files to avoid excessive `openat` calls
    report_to="none",
    push_to_hub=False,
    fp16=True,  # Mixed Precision Training for speedup
)

# Trainer with optimized dataloader
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Run training with optimizations
if args.profile:
    print("Running training with PyTorch Profiler...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs/profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as profiler:
        for step, batch in enumerate(trainer.get_train_dataloader()):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.cuda.stream(torch.cuda.Stream()):  # Optimize `ioctl()` and GPU execution
                outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            profiler.step()
            if step >= 5:
                break
        trainer.train()
else:
    print("Running training without profiler...")
    trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Save the model
model.save_pretrained("./bert_sentiment_model")
tokenizer.save_pretrained("./bert_sentiment_model")
