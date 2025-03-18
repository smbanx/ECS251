# Basic Structure of our code

[Pytorch Syscall Comparison](https://github.com/smbanx/ECS251/tree/main/ML_Syscall_comparison) 

[Code for DataLoader IO_Uring Implementation](https://github.com/smbanx/ECS251/tree/main/dataloader_io_uring)

[Code & Visualization for IO_Uring Save Model Implementation](https://github.com/smbanx/ECS251/tree/main/iouring_save_model)

[Output for Strace and Pytorch Syscall](https://github.com/smbanx/ECS251/tree/main/output)

[Tools to parse results from profiler](https://github.com/smbanx/ECS251/tree/main/tools)

[Attention Based Workload](https://github.com/smbanx/ECS251/tree/main/workload)

[History of Profiling Commands ran](https://github.com/smbanx/ECS251/blob/main/history.txt)

# Concepts from your project plan map to that code

Our project consisted of 2 parts. The first part is profiling the workload. The [workload](https://github.com/smbanx/ECS251/blob/main/workload/ECS251.py) which is fine-tuning an attention based model for sentiment analysis. We ran the commands in [History of Profiling Commands ran](https://github.com/smbanx/ECS251/blob/main/history.txt) for the traces. The results for the traces are in [Output for Strace and Pytorch Syscall](https://github.com/smbanx/ECS251/tree/main/output). Then we parsed and visualized the result using [Tools to parse results from profiler](https://github.com/smbanx/ECS251/tree/main/tools). After we visualized the results we decided to explore various ways to reduce syscall overhead.

Part 2 involves in trying out ways to reduce I/O overhead within our workload. We tried to reduce I/O operations on the write model to disk part in [Code & Visualization for IO_Uring Save Model Implementation](https://github.com/smbanx/ECS251/tree/main/iouring_save_model). We also tried to reduce I/O overhead during the data loading phrase in [Code for DataLoader IO_Uring Implementation](https://github.com/smbanx/ECS251/tree/main/dataloader_io_uring). Finally we tried to optimize for commands like [clock_gettime, ioctl, etc](https://github.com/smbanx/ECS251/blob/main/workload/ECS251_optimized.py).

# Profiling commands

### Strace profiling commands
strace -o strace_output.txt -f python \<INSERT FILE HERE\>

py-spy record --output py_spy_profile.svg -- python \<INSERT FILE HERE\>

strace -c -o strace_summary.txt python \<INSERT FILE HERE\>

#### Strace profiling on individual ML parts
strace -o strace_output.txt -f python workload/trace_syscalls.py \<--trace_copy/--trace_forward/--trace_backward\> --mode \<0 / 1\>


### Pytorch profiling commands
python \<INSERT FILE HERE\> --profile
