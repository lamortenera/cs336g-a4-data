import argparse
import gzip
from random import random
from cs336_data.sample_urls import count_lines
from fastwarc.warc import ArchiveIterator, WarcRecordType

parser = argparse.ArgumentParser()
parser.add_argument(
    "--positive_warc", type=str, help="Path to the positive .warc.gz file"
)
parser.add_argument(
    "negative_warc", type=str, help="Path to the negative .warc.gz file"
)
parser.add_argument(
    "--max_per_class", type=int, help="Maximum number of samples to include per class",
    default=10000
)
parser.add_argument(
    "--train_output", type=str, help="Path to the output training data file"
)
parser.add_argument(
    "--test_output", type=str, help="Path to the output test data file"
)

def record_generator(file_path):
    """Generator that yields records from a .warc.gz file."""
    with open(file_path, "rb") as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            yield record

def count_records(file_path):
    """Count total records in a .warc.gz file."""
    for i, _ in enumerate(record_generator(file_path)):
        pass
    return i + 1

def sample_record(positive_path, negative_path, positive_prob, positive_rate, negative_rate):
    positive_gen = record_generator(positive_path)
    negative_gen = record_generator(negative_path)
    while True:
        if random.random() < positive_prob:
            try:
                record = next(positive_gen)
                if random.random() < positive_rate:
                    yield "positive", record
            except StopIteration:
                break
        else:
            try:
                record = next(negative_gen)
                if random.random() < negative_rate:
                    yield "negative", record
            except StopIteration:
                break

def main():
    args = parser.parse_args()

    # Count total records in both files
    positive_records = count_records(args.positive_warc)
    negative_records = count_records(args.negative_warc)

    print(f"Total records in positive file: {positive_records}")
    print(f"Total records in negative file: {negative_records}")

    capped_positive_records = min(positive_records, args.max_per_class)
    capped_negative_records = min(negative_records, args.max_per_class)
    positive_rate = capped_positive_records / positive_records
    negative_rate = capped_negative_records / negative_records
    positive_prob = capped_positive_records / (capped_positive_records + capped_negative_records)


    print(f"Sampling with positive rate: {positive_rate:.4f}, negative rate: {negative_rate:.4f}, positive probability: {positive_prob:.4f}")

    sampling_stats = {
        "test_samples": {"positive": 0, "negative": 0},
        "train_samples": {"positive": 0, "negative": 0}
    }    
    with open(args.train_output, "w", encoding="utf-8") as train_file, \
         open(args.test_output, "w", encoding="utf-8") as test_file:
        for label, record in sample_record(args.positive_warc, args.negative_warc, positive_prob, positive_rate, negative_rate):
            content = record.reader.read()
            line = f"__label__{label}\n{content.decode('utf-8', errors='ignore').replace('\n', ' ')}\n"
            if random.random() < 0.9:
                train_file.write(line)
                sampling_stats["train_samples"][label] += 1
            else:
                test_file.write(line)
                sampling_stats["test_samples"][label] += 1
            
    print(f"Sampled {sampling_stats['train_samples']['positive']} positive and {sampling_stats['train_samples']['negative']} negative records in training set.")
    print(f"Sampled {sampling_stats['test_samples']['positive']} positive and {sampling_stats['test_samples']['negative']} negative records in test set.")