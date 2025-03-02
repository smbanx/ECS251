import re

def clean_log_file(file_path):
    """Reads and cleans a log file by removing process IDs and hexadecimal addresses."""
    cleaned_lines = set()
    
    with open(file_path, 'r') as file:
        for line in file:
            # Remove the first number (process ID)
            line = re.sub(r'^\d+\s+', '', line)
            # Remove hexadecimal addresses
            line = re.sub(r'0x[0-9a-fA-F]+', '', line)
            # Strip extra spaces that may appear
            line = line.strip()
            if line:
                cleaned_lines.add(line)
    
    return cleaned_lines

def compare_logs(file1, file2, output_unique1, output_unique2):
    """Compares two cleaned log files and saves unique lines from each file separately."""
    cleaned1 = clean_log_file(file1)
    cleaned2 = clean_log_file(file2)

    # Find unique lines in each file
    unique_to_file1 = cleaned1 - cleaned2
    unique_to_file2 = cleaned2 - cleaned1

    # Save unique lines from file1
    with open(output_unique1, 'w') as out_file1:
        for line in sorted(unique_to_file1):
            out_file1.write(line + '\n')

    # Save unique lines from file2
    with open(output_unique2, 'w') as out_file2:
        for line in sorted(unique_to_file2):
            out_file2.write(line + '\n')

# Example usage:
file1 = "iouring.log"
file2 = "original.log"
output_unique1 = "iouring_c.log"
output_unique2 = "original_c.log"

compare_logs(file1, file2, output_unique1, output_unique2)
