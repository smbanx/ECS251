import torch
import time
import argparse
from torch.profiler import profile, record_function, ProfilerActivity

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Trace PyTorch syscalls with torch.profiler.")
parser.add_argument("--trace_copy", action="store_true", help="Trace memory copy syscalls (Aten::to, Aten::_to_copy, Aten::copy_)")
parser.add_argument("--trace_forward", action="store_true", help="Trace forward pass syscalls (Aten::linear, Aten::addmm)")
parser.add_argument("--trace_backward", action="store_true", help="Trace backward pass syscalls (Aten::mm, cudaLaunchKernel, Aten::add_)")
parser.add_argument("--mode", type=int, choices=[0, 1], default=1, help="Mode 0: Baseline (no computation), Mode 1: Full execution")
args = parser.parse_args()

# Generate trace filename dynamically based on selected options
trace_types = []
if args.trace_copy:
    trace_types.append("copy")
if args.trace_forward:
    trace_types.append("forward")
if args.trace_backward:
    trace_types.append("backward")

# Default to "trace_default" if no options are specified
trace_filename = f"trace_{'_'.join(trace_types)}.json" if trace_types else "trace_default.json"

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)

# Create input tensor
x = torch.randn(5, 10, device="cpu", requires_grad=True)
w = torch.randn(20, 10, device="cpu", requires_grad=True)
b = torch.randn(20, device="cpu", requires_grad=True)

if (args.trace_forward or args.trace_backward) and torch.cuda.is_available():
    with record_function("Copying tensors to GPU"):
            x = x.to(device)
            w = w.to(device)
            b = b.to(device)
    torch.cuda.synchronize()

if args.trace_backward:
    with record_function("Forward Pass"):
            y = torch.nn.functional.linear(x, w, b)

# Define Profiler
'''with profile(
    activities=[
        ProfilerActivity.CPU, 
        ProfilerActivity.CUDA if torch.cuda.is_available() else None
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
'''
if args.mode == 1:  # Run computation only in mode 1
# Copy to GPU (if available) - Traces Aten::to, Aten::_to_copy, Aten::copy_, cudaStreamSynchronize
    if args.trace_copy and torch.cuda.is_available():
        with record_function("Copying tensors to GPU"):
            x = x.to(device)
            w = w.to(device)
            b = b.to(device)
        torch.cuda.synchronize()  # Equivalent to cudaStreamSynchronize()

    # Forward Pass - Traces Aten::linear, Aten::addmm
    if args.trace_forward:
        with record_function("Forward Pass"):
            y = torch.nn.functional.linear(x, w, b)

    # Backward Pass - Traces Aten::mm, cudaLaunchKernel, Aten::add_
    if args.trace_backward:
        with record_function("Backward Pass"):
            y.sum().backward()

# Print Profiler Output
#print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

# Save the profiling result dynamically
#prof.export_chrome_trace(trace_filename)

print(f"Tracing complete. Open `{trace_filename}` in Chrome (chrome://tracing) to visualize.")
