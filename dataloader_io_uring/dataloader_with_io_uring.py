# -*- coding: utf-8 -*-

!pip install numpy
!pip install datasets
!pip install transformers
!pip install scikit-learn
!pip install evaluate
!pip install git+https://github.com/YoSTEALTH/Liburing.git

import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from liburing import (
    O_RDONLY, AT_FDCWD, iovec, io_uring, io_uring_get_sqe,
    io_uring_prep_openat, io_uring_prep_read, io_uring_prep_close,
    io_uring_submit, io_uring_wait_cqe, io_uring_cqe_seen,
    io_uring_cqe, io_uring_queue_init, io_uring_queue_exit,
    io_uring_sqe_set_data64,io_uring_prep_readv, trap_error
)
from sklearn.metrics import accuracy_score, f1_score
import json
from tqdm import tqdm

# Configuration class with reduced dataset size and memory footprint
class BenchmarkConfig:
    """Configuration for the benchmark"""
    def __init__(self,
                 data_dir="./imdb_data",
                 cache_dir="./cache",
                 batch_size=8,  # Reduced batch size
                 num_workers=2,  # Reduced workers
                 queue_depth=64,  # Reduced queue depth
                 num_epochs=2,    # Reduced epochs
                 max_length=128,  # Reduced max length
                 learning_rate=2e-5,
                 sample_size=100, # Significantly reduced sample size
                 test_size=20,    # Reduced test size
                 seed=42):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.queue_depth = queue_depth
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.sample_size = sample_size
        self.test_size = test_size
        self.seed = seed

        # Create output directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)


# Fixed helper class for io_uring operations
class IoUringHelper:
    """Helper class for io_uring operations"""
    def __init__(self, queue_depth=64):
        self.ring = io_uring()
        self.cqe = io_uring_cqe()
        self.queue_depth = queue_depth
        io_uring_queue_init(queue_depth, self.ring, 0)

    def __del__(self):
        """Clean up resources"""
        try:
            io_uring_queue_exit(self.ring)
        except:
            pass

    

    def open_file(self, file_path):
        """Open a file using io_uring"""
        if isinstance(file_path, str):
            _path = file_path.encode()  # Ensure bytes format
        elif isinstance(file_path, bytes):
            _path = file_path
        else:
            raise TypeError(f"Invalid file path type: {type(file_path)}. Expected str or bytes.")

        sqe = io_uring_get_sqe(self.ring)  # Keeping original method of getting SQE

        io_uring_prep_openat(sqe, _path, O_RDONLY, 0o777, AT_FDCWD)  # Keeping dfd parameter

        io_uring_sqe_set_data64(sqe, 1)

        return self._submit_and_wait()


    def read_file(self, fd, file_size):
        """Reads a file using io_uring"""
        if file_size <= 0:
            raise ValueError("Invalid file size for reading.")

        buf = bytearray(file_size)  # Allocate buffer
        iov = iovec(buf)  # Wrap it in an iovec structure

        sqe = io_uring_get_sqe(self.ring)
        io_uring_prep_readv(sqe, fd, iov, 0)  # Pass only 4 arguments

        io_uring_sqe_set_data64(sqe, 1)
        self._submit_and_wait()

        return bytes(buf)  # Convert bytearray to immutable bytes before returning




    def close_file(self, fd):
        """Close a file using io_uring"""
        sqe = io_uring_get_sqe(self.ring)
        io_uring_prep_close(sqe, fd)
        io_uring_sqe_set_data64(sqe, 3)
        self._submit_and_wait()

    def _submit_and_wait(self):
        """Submit operation and wait for completion"""
        io_uring_submit(self.ring)
        io_uring_wait_cqe(self.ring, self.cqe)
        result = trap_error(self.cqe.res)
        io_uring_cqe_seen(self.ring, self.cqe)
        return result


# Prepare IMDB dataset and save as individual files
def prepare_imdb_dataset(config):
    """Download and prepare IMDB dataset, saving each review as a separate file"""
    print("Loading IMDB dataset...")
    # Load the dataset
    dataset = load_dataset("imdb", cache_dir=config.cache_dir)

    # Create train and test directories
    train_dir = os.path.join(config.data_dir, "train")
    test_dir = os.path.join(config.data_dir, "test")

    # Clean existing files to avoid conflicts
    for directory in [train_dir, test_dir]:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Prepare training set (reduced sample size)
    print(f"Preparing training subset ({config.sample_size} samples)...")
    train_data = dataset["train"].shuffle(seed=config.seed).select(range(config.sample_size))

    for i, item in enumerate(tqdm(train_data)):
        # Save text and label
        file_path = os.path.join(train_dir, f"train_{i}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"text": item["text"], "label": item["label"]}, f)

    # Prepare test set (smaller subset for quick validation)
    print(f"Preparing test subset ({config.test_size} samples)...")
    test_data = dataset["test"].shuffle(seed=config.seed).select(range(config.test_size))

    for i, item in enumerate(tqdm(test_data)):
        file_path = os.path.join(test_dir, f"test_{i}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"text": item["text"], "label": item["label"]}, f)

    print(f"Dataset preparation complete. Files stored in {config.data_dir}")
    return {"train": train_dir, "test": test_dir}


# Standard PyTorch Dataset for IMDB
class VanillaIMDBDataset(Dataset):
    """Standard PyTorch dataset using regular file I/O for IMDB data"""
    def __init__(self, data_dir, tokenizer, max_length=128):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.json')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load file using regular I/O
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Get text and label
            text = data['text']
            label = data['label']

            # Tokenize text
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Return the encoded tokens and label
            return {
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            # Return empty tensors in case of error
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'label': torch.tensor(0, dtype=torch.long)
            }


# Fixed Dataset implementation using io_uring
class IoUringIMDBDataset(Dataset):
    """Dataset implementation using io_uring for async I/O for IMDB data"""
    def __init__(self, data_dir, tokenizer, max_length=128, queue_depth=64):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        self.queue_depth = queue_depth
        self.io_helper = None
        # self.io_helper = IoUringHelper(self.queue_depth)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.io_helper is None:
            self.io_helper = IoUringHelper(self.queue_depth)

        # Get file path
        file_path = os.path.join(self.data_dir, self.file_list[idx])

        # Open file using io_uring
        fd = self.io_helper.open_file(file_path)
        if fd < 0:
            raise IOError(f"Failed to open file: {file_path}, error code: {fd}")

        # Get file size
        file_size = os.path.getsize(file_path)

        # Read file content using io_uring
        content_bytes = self.io_helper.read_file(fd, file_size)

        # Close the file
        self.io_helper.close_file(fd)

        # Decode and parse JSON
        content_str = content_bytes.decode('utf-8')
        data = json.loads(content_str)

        # Get text and label
        text = data['text']
        label = data['label']

        # Tokenize text
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }



# Training function for BERT on IMDB
def train_model(config, dataloader, test_dataloader, model, optimizer, scheduler, device, desc=""):
    """Train BERT model on IMDB dataset using specified dataloader"""
    start_time = time.time()

    # Track metrics
    epoch_times = []
    batch_io_times = []
    batch_compute_times = []
    train_losses = []

    # Training loop
    print(f"\nTraining with {desc}...")

    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0

        # Use tqdm for progress bar
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        batch_start = time.time()

        for batch_idx, batch in enumerate(loop):
            # Track I/O time (time to get the batch)
            batch_loaded = time.time()
            io_time = batch_loaded - batch_start
            batch_io_times.append(io_time)

            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Track compute time
            batch_end = time.time()
            compute_time = batch_end - batch_loaded
            batch_compute_times.append(compute_time)

            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            # Update progress bar
            loop.set_postfix(loss=avg_loss)

            # Reset for next batch
            batch_start = time.time()

        # Track epoch time
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)

        # Save epoch loss
        train_losses.append(total_loss / len(dataloader))

        # Evaluate on test set
        val_metrics = evaluate_model(test_dataloader, model, device)
        print(f"Epoch {epoch+1} - Loss: {train_losses[-1]:.4f}, Validation Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

    # Final evaluation
    final_metrics = evaluate_model(test_dataloader, model, device)

    # Calculate timing statistics
    total_time = time.time() - start_time
    avg_epoch_time = np.mean(epoch_times)
    avg_io_time = np.mean(batch_io_times)
    avg_compute_time = np.mean(batch_compute_times)
    io_percentage = (avg_io_time / (avg_io_time + avg_compute_time)) * 100

    results = {
        'total_time': total_time,
        'avg_epoch_time': avg_epoch_time,
        'avg_io_time': avg_io_time,
        'avg_compute_time': avg_compute_time,
        'io_percentage': io_percentage,
        'train_losses': train_losses,
        'final_accuracy': final_metrics['accuracy'],
        'final_f1': final_metrics['f1']
    }

    # Print summary
    print(f"\nTraining completed with {desc}:")
    print(f"  Total training time: {total_time:.2f}s")
    print(f"  Average epoch time: {avg_epoch_time:.2f}s")
    print(f"  Average batch I/O time: {avg_io_time:.4f}s")
    print(f"  Average batch compute time: {avg_compute_time:.4f}s")
    print(f"  I/O percentage of batch time: {io_percentage:.2f}%")
    print(f"  Final validation accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Final validation F1 score: {final_metrics['f1']:.4f}")

    return results


# Evaluation function
def evaluate_model(dataloader, model, device):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')

    return {
        'accuracy': accuracy,
        'f1': f1
    }


# Compare results function
def compare_results(vanilla_results, iouring_results):
    """Compare and summarize results between vanilla and io_uring implementations"""
    # Calculate improvements
    time_improvement = (vanilla_results['total_time'] - iouring_results['total_time']) / vanilla_results['total_time'] * 100
    io_time_improvement = (vanilla_results['avg_io_time'] - iouring_results['avg_io_time']) / vanilla_results['avg_io_time'] * 100
    epoch_time_improvement = (vanilla_results['avg_epoch_time'] - iouring_results['avg_epoch_time']) / vanilla_results['avg_epoch_time'] * 100

    # Print comparison
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*50)

    print("\nTime Metrics:")
    print(f"  Total training time: {vanilla_results['total_time']:.2f}s vs {iouring_results['total_time']:.2f}s ({time_improvement:.2f}% improvement)")
    print(f"  Average epoch time: {vanilla_results['avg_epoch_time']:.2f}s vs {iouring_results['avg_epoch_time']:.2f}s ({epoch_time_improvement:.2f}% improvement)")
    print(f"  Average I/O time: {vanilla_results['avg_io_time']:.4f}s vs {iouring_results['avg_io_time']:.4f}s ({io_time_improvement:.2f}% improvement)")

    print("\nI/O Bottleneck Analysis:")
    print(f"  Vanilla I/O percentage: {vanilla_results['io_percentage']:.2f}%")
    print(f"  io_uring I/O percentage: {iouring_results['io_percentage']:.2f}%")

    print("\nPerformance Metrics:")
    print(f"  Vanilla final accuracy: {vanilla_results['final_accuracy']:.4f}")
    print(f"  io_uring final accuracy: {iouring_results['final_accuracy']:.4f}")
    print(f"  Vanilla final F1 score: {vanilla_results['final_f1']:.4f}")
    print(f"  io_uring final F1 score: {iouring_results['final_f1']:.4f}")

    # Calculate throughput (samples per second)
    vanilla_throughput = config.sample_size * config.num_epochs / vanilla_results['total_time']
    iouring_throughput = config.sample_size * config.num_epochs / iouring_results['total_time']
    throughput_improvement = (iouring_throughput - vanilla_throughput) / vanilla_throughput * 100

    print("\nThroughput Analysis:")
    print(f"  Vanilla throughput: {vanilla_throughput:.2f} samples/second")
    print(f"  io_uring throughput: {iouring_throughput:.2f} samples/second")
    print(f"  Throughput improvement: {throughput_improvement:.2f}%")

    # Overall assessment
    print("\nOverall Assessment:")
    if io_time_improvement > 5:
        print(f"  io_uring provides significant I/O performance improvement ({io_time_improvement:.2f}%)")
    elif io_time_improvement > 0:
        print(f"  io_uring provides modest I/O performance improvement ({io_time_improvement:.2f}%)")
    else:
        print(f"  io_uring does not provide I/O performance improvement ({io_time_improvement:.2f}%)")

    if time_improvement > 5:
        print(f"  Overall training time improved significantly ({time_improvement:.2f}%)")
    elif time_improvement > 0:
        print(f"  Overall training time improved modestly ({time_improvement:.2f}%)")
    else:
        print(f"  No improvement in overall training time ({time_improvement:.2f}%)")

    # Conclusion
    print("\nConclusion:")
    io_percentage_diff = vanilla_results['io_percentage'] - iouring_results['io_percentage']
    if io_percentage_diff > 5:
        print(f"  io_uring reduced I/O bottleneck by {io_percentage_diff:.2f} percentage points")

    if vanilla_results['io_percentage'] < 10:
        print("  The workload is not I/O bound (I/O < 10% of processing time)")
        print("  For compute-bound workloads, I/O optimizations have limited impact")

    if throughput_improvement > 0:
        print(f"  Using io_uring improved training throughput by {throughput_improvement:.2f}%")
    else:
        print(f"  Using io_uring did not improve training throughput ({throughput_improvement:.2f}%)")


# Main execution function
def run_imdb_comparison():
    """Main function to run the IMDB comparison benchmark"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Prepare dataset (force recreation to ensure clean state)
    data_paths = prepare_imdb_dataset(config)

    # Create datasets
    print("Creating datasets...")
    vanilla_train_dataset = VanillaIMDBDataset(data_paths["train"], tokenizer, max_length=config.max_length)
    vanilla_test_dataset = VanillaIMDBDataset(data_paths["test"], tokenizer, max_length=config.max_length)

    iouring_train_dataset = IoUringIMDBDataset(data_paths["train"], tokenizer, max_length=config.max_length, queue_depth=config.queue_depth)
    iouring_test_dataset = IoUringIMDBDataset(data_paths["test"], tokenizer, max_length=config.max_length, queue_depth=config.queue_depth)

    # Create dataloaders
    print("Creating dataloaders...")
    vanilla_train_dataloader = DataLoader(
        vanilla_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda")
    )

    vanilla_test_dataloader = DataLoader(
        vanilla_test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda")
    )

    iouring_train_dataloader = DataLoader(
        iouring_train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda")
    )

    iouring_test_dataloader = DataLoader(
        iouring_test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # Train with vanilla dataloader
    print("\nInitializing BERT model for vanilla training...")
    vanilla_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    vanilla_model.to(device)

    # Initialize optimizer and scheduler
    vanilla_optimizer = AdamW(vanilla_model.parameters(), lr=config.learning_rate, eps=1e-8)
    vanilla_total_steps = len(vanilla_train_dataloader) * config.num_epochs
    vanilla_scheduler = get_linear_schedule_with_warmup(
        vanilla_optimizer,
        num_warmup_steps=0,
        num_training_steps=vanilla_total_steps
    )

    # Train with vanilla dataloader
    vanilla_results = train_model(
        config,
        vanilla_train_dataloader,
        vanilla_test_dataloader,
        vanilla_model,
        vanilla_optimizer,
        vanilla_scheduler,
        device,
        desc="Vanilla DataLoader"
    )

    # Train with io_uring dataloader
    print("\nInitializing BERT model for io_uring training...")
    iouring_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    iouring_model.to(device)

    # Initialize optimizer and scheduler
    iouring_optimizer = AdamW(iouring_model.parameters(), lr=config.learning_rate, eps=1e-8)
    iouring_total_steps = len(iouring_train_dataloader) * config.num_epochs
    iouring_scheduler = get_linear_schedule_with_warmup(
        iouring_optimizer,
        num_warmup_steps=0,
        num_training_steps=iouring_total_steps
    )

    # Train with io_uring dataloader
    iouring_results = train_model(
        config,
        iouring_train_dataloader,
        iouring_test_dataloader,
        iouring_model,
        iouring_optimizer,
        iouring_scheduler,
        device,
        desc="io_uring DataLoader"
    )

    # Compare results
    compare_results(vanilla_results, iouring_results)


# Initialize benchmark configuration with reduced settings
config = BenchmarkConfig(
    data_dir="./imdb_data",
    cache_dir="./cache",
    batch_size=8,        # Reduced batch size
    num_workers=2,       # Reduced workers
    queue_depth=64,      # Reduced queue depth
    num_epochs=2,        # Reduced epochs
    max_length=128,      # Reduced max sequence length
    learning_rate=2e-5,
    sample_size=20000,     # Significantly reduced sample size
    test_size=5000,        # Reduced test size
    seed=42
)

# Run the benchmark
if __name__ == "__main__":
    # This allows the code to be executed directly or imported
    run_imdb_comparison()
else:
    # When run in a notebook, execute this
    print("Ready to run IMDB io_uring benchmark. Execute run_imdb_comparison() to start.")