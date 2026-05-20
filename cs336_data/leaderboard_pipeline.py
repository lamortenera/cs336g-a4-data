"""Processes WARC records for the pipeline."""

from fastwarc.warc import ArchiveIterator, WarcRecord, WarcRecordType, WarcWriter

import argparse

import assignment
import pathlib
from collections import defaultdict
import random
import io
import json
from collections.abc import Callable, Generator


parser = argparse.ArgumentParser()
parser.add_argument("input", help="the input WARC WET file")
parser.add_argument("output_dir", help="the output directory for filtered records")

def increase_counter(d, key):
    d[key] = d.get(key, 0) + 1

def append_example(d, key, example, max_examples=20):
    curr_examples = d.get(key, [])
    curr_examples.insert(example, random.randint(0, len(curr_examples)))
    del curr_examples[max_examples:]
    d[key] = curr_examples

def modify_and_write_record(original_record, new_content_bytes, writer):
    """
    Takes an existing fastwarc record, replaces its body with new_content_bytes,
    and writes it to a gzipped WARC file using a fastwarc writer.
    """
    # 1. Convert your new content into a file-like bytes stream
    new_body_stream = io.BytesIO(new_content_bytes)
    
    # 2. Extract and update the headers from the original record
    # Fastwarc headers can be converted directly into a dictionary
    new_headers = dict(original_record.headers)
    
    # 3. Update the Content-Length header to match the new body size
    new_headers['Content-Length'] = str(len(new_content_bytes))
    
    # 4. Create the new record object
    # Pass the original record type (e.g., WarcRecordType.response)
    modified_record = WarcRecord(
        headers=new_headers,
        payload=new_body_stream,
        record_type=original_record.record_type
    )
    
    # 5. Write the record (fastwarc handles the gzipping if the writer is configured for it)
    writer.write(modified_record)

def filter_records(input: pathlib.Path, output: pathlib.Path):
    stats = {}
    with input.open("rb") as f_in, output.open("wb") as f_out:
        writer = WarcWriter(f_out, gzip=True)
        for record in ArchiveIterator(f_in, record_types=WarcRecordType.response):
            text = record.reader.read().decode("utf8")
            language, _ = assignment.identify_language(text)
            increase_counter(stats, "processed")
            if language != "en":
                increase_counter(stats, "filtered_language")
                continue
            if not assignment.gopher_quality_filter(text):
                increase_counter(stats, "filtered_gopher")
                append_example(stats, "x_gopher", text)
                continue
            normalized_text = text.replace("\n", " ")
            nsfw_label, _ = assignment.classify_nsfw(normalized_text, normalized=True)
            if nsfw_label == "nsfw":
                increase_counter(stats, "filtered_nsfw")
                append_example(stats, "x_nsfw", text)
                continue
            toxicity_label, _ = assignment.classify_hate_speech(normalized_text, normalized=True)
            if toxicity_label == "toxic":
                increase_counter(stats, "filtered_toxic")
                append_example(stats, "x_toxic", text)
                continue
            good_qual, _ = assignment.classify_quality(normalized_text, normalized=True)
            if not good_qual:
                increase_counter(stats, "filtered_lowqual")
                append_example(stats, "x_lowqual", text)
                continue
            increase_counter(stats, "processed_valid")
            masked_text, _ = assignment.mask_emails(text)
            masked_text, _ = assignment.mask_phone_numbers(masked_text)
            masked_text, _ = assignment.mask_ips(masked_text)
            modify_and_write_record(record, masked_text.encode("utf8"), writer)


def warc_records(input_files: list[pathlib.Path]) -> Generator[tuple[WarcRecord, str], None, None]:
    for input_file in input_files:
        with input_file.open("rb") as f_in:
            for record in ArchiveIterator(f_in, record_types=WarcRecordType.response):
                yield record, record.reader.read().decode("utf8")


def remove_dup_lines(input_files: list[pathlib.Path], output_file: pathlib.Path):
    input_records = lambda: warc_records(input_files)
    stats = {}
    filtered_records = assignment.exact_line_dedup_records(input_records)
    with output_file.open("wb") as f_out:
        writer = WarcWriter(f_out, gzip=True)
        for record, filtered_text in filtered_records:
            increase_counter(stats, "line_filtering_before")
            if not filtered_text.strip():
                increase_counter(stats, "line_filtering_after")
                continue
            modify_and_write_record(record, filtered_text.encode("utf8"), writer)
    return stats


def remove_neardups(input_files: list[pathlib.Path], output_file: pathlib.Path):
    input_records = lambda: warc_records(input_files)
    stats = {}
    filtered_records = assignment.minhash_deduplication_from_records(
        input_records, num_hashes=100, num_bands=10, ngram_len=4, jaccard_threshold=0.8
    )

    with output_file.open("wb") as f_out:
        writer = WarcWriter(f_out, gzip=True)
        for record in filtered_records:
            increase_counter(stats, "neardup_filtering_after")
            writer.write(record)


if __name__ == "__main__":
    args = parser.parse_args()
    input_path = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output_dir)
    output_path = output_dir / "filtered_records.warc.wet.gz"
    stats_json = output_dir / "filtering_stats.json"
    stats = filter_records(input_path, output_path)
    with stats_json.open("w") as json_out:
        json.dump(stats, json_out)
    
    filtered_dup_lines = output_dir / "filtered_dup_lines.warc.wet.gz"
    exact_dupes_stats = remove_dup_lines([output_path], filtered_dup_lines)
    stats.update(exact_dupes_stats)

    jaccard_filtered = output_dir / "filtered_jaccard.warc.wet.gz"
    neardup_stats = remove_neardups([filtered_dup_lines], jaccard_filtered)
    
    stats.update(neardup_stats)

    for key, val in stats.items():
        if not isinstance(val, list):
            print(f"{key}: {val}")
    
    