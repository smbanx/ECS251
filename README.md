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

To run strace profiler on any workload and put the results into the output file "strace_output.txt"

```
strace -o strace_output.txt -f python <INSERT FILE HERE>
```

To generate a flame graph of execution

```
py-spy record --output py_spy_profile.svg -- python <INSERT FILE HERE>
```

To generate a strace summary

```
strace -c -o strace_summary.txt python <INSERT FILE HERE>
```

#### Strace profiling on individual ML parts
```
strace -o strace_output.txt -f python workload/trace_syscalls.py <--trace_copy OR --trace_forward OR --trace_backward> --mode <0 OR 1>
```




### Pytorch profiling commands
```
python workload/ECS251.py --profile
```

# What the tools do

[Strace Visualization](https://github.com/smbanx/ECS251/blob/main/tools/Strace_Visualization.ipynb) run this file with strace results to generate pie graph and syscall by time visualizations

[Strace Parser](strace_parser) Parse Strace results into .json file with PID, Syscall, Arguments, and Return Value for further processing.

[Clean Logs](https://github.com/smbanx/ECS251/blob/main/tools/clean_logs.py) filter the same strace logs from two different strace outputs so that we can work on different syscalls.




