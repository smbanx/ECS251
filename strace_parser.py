import sys
import json
import re

def parse(data):
    # Regular expression pattern to parse strace lines
    strace_pattern = re.compile(r"""
        (?P<pid>\d+)\s+  # Process ID
        (?P<syscall>\w+)\((?P<args>.*?)\)  # System call and arguments inside parentheses
        \s*=\s*
        (?P<retval>[-\w\s\(\)\/x]+)  # Return value, possibly with an error message
    """, re.VERBOSE)

    # Parse each line and store results
    parsed_calls = []
    for line in data.strip().split("\n"):
        match = strace_pattern.match(line)
        if match:
            parsed_calls.append({
                "PID": match.group("pid"),
                "Syscall": match.group("syscall"),
                "Arguments": match.group("args"),
                "Return Value": match.group("retval")
            })
    return parsed_calls

def read_file(filename):
    """Reads a file and returns its contents as a string."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except IOError:
        print(f"Error: Unable to read the file '{filename}'.")

def save_as_json(data, filename):
    """Saves the given data as a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        print(f"Data successfully saved to {filename}")
    except IOError:
        print(f"Error: Unable to write to file '{filename}'.")


def main():
    """Main function to handle command-line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    content = read_file(filename)
    parsed_data = parse(content)
    output_filename = "parsed_" + filename + ".json"
    save_as_json(parsed_data, output_filename)

if __name__ == "__main__":
    main()