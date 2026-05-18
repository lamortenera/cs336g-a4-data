import argparse
import gzip
import random
import sys


def count_lines(file_path):
    """Count total lines in a gzip file using a fast chunked read loop."""
    line_count = 0
    with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
        # Reading in chunks is significantly faster than standard line iteration
        for chunk in f:
            line_count += chunk.count("\n")
    return line_count

def stream_sample_gz(file_path, n, total_lines):
    """Stream lines and print selected ones based on the calculated sample rate."""
    if total_lines == 0 or n <= 0:
        return

    # Calculate exact sampling probability per line
    sample_rate = min(1.0, n / total_lines)

    # Seed the random number generator for consistency if needed
    random.seed(123)

    num_sampled = 0

    with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if random.random() < sample_rate:
                # Print directly to stdout without adding extra newlines
                sys.stdout.write(line)
                num_sampled += 1

    print(f"Sampled {num_sampled} lines out of {total_lines} total lines.", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="Sample lines from a .gz file based on target sample size."
    )
    parser.add_argument(
        "file_path", type=str, help="Path to the input .txt.gz file"
    )
    parser.add_argument(
        "n", type=int, help="Target number of lines to sample (n)"
    )

    args = parser.parse_args()

    # Step 1: Count total lines (N)
    total_lines = count_lines(args.file_path)
    print(f"Total lines in file: {total_lines}", file=sys.stderr)

    # Step 2: Stream, sample with probability n/N, and print
    stream_sample_gz(args.file_path, args.n, total_lines)


if __name__ == "__main__":
    main()
