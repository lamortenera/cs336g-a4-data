"""Processes WARC records for the pipeline."""

import shutil
from time import time

from fastwarc.warc import ArchiveIterator, WarcRecord, WarcRecordType
from fastwarc.stream_io import GZipStream

import argparse

import assignment
import pathlib
from collections import defaultdict
import random
import io
import json
from collections.abc import Callable, Generator
from cs336_data.wet import get_wet_file_urls, download_wet_file
import concurrent.futures
import os
from tqdm import tqdm
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("input", help="the input WARC WET file")
parser.add_argument("output_dir", help="the output directory for filtered records")
parser.add_argument("--max_urls", type=int, help="Maximum number of URLs to process if using --from_wet_urls", default=10)

def increase_counter(d, key):
    d[key] = d.get(key, 0) + 1

def append_example(d, key, example, max_examples=20):
    curr_examples = d.get(key, [])
    curr_examples.insert(random.randint(0, len(curr_examples)), example)
    del curr_examples[max_examples:]
    d[key] = curr_examples

def modify_and_write_record(original_record, new_content_bytes, writer):
    """
    Takes an existing fastwarc record, replaces its body with new_content_bytes,
    and writes it to a gzipped WARC file using a fastwarc writer.
    """
    original_record.set_bytes_content(new_content_bytes)
    original_record.write(writer)

def filter_records(input: pathlib.Path, output: pathlib.Path):
    stats = {}
    with input.open("rb") as f_in, output.open("wb") as f_out:
        writer = GZipStream(f_out)
        for record in ArchiveIterator(f_in, record_types=WarcRecordType.conversion):
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
    return stats


def warc_records(input_files: list[pathlib.Path]) -> Generator[tuple[WarcRecord, str], None, None]:
    for input_file in input_files:
        with input_file.open("rb") as f_in:
            for record in ArchiveIterator(f_in, record_types=WarcRecordType.conversion):
                yield record, record.reader.read().decode("utf8")


def remove_dup_lines(input_files: list[pathlib.Path], output_file: pathlib.Path):
    input_records = lambda: warc_records(input_files)
    stats = {}
    filtered_records = assignment.exact_line_dedup_records(input_records)
    with output_file.open("wb") as f_out:
        writer = GZipStream(f_out)
        for record, filtered_text in filtered_records:
            increase_counter(stats, "line_filtering_before")
            if not filtered_text.strip():
                continue
            increase_counter(stats, "line_filtering_after")
            modify_and_write_record(record, filtered_text.encode("utf8"), writer)
    return stats


def remove_neardups(input_files: list[pathlib.Path], output_file: pathlib.Path):
    input_records = lambda: warc_records(input_files)
    stats = {}
    filtered_records = assignment.minhash_deduplication_from_records(
        input_records, num_hashes=100, num_bands=10, ngram_len=4, jaccard_threshold=0.8
    )

    with output_file.open("wb") as f_out:
        writer = GZipStream(f_out)
        for record in filtered_records:
            increase_counter(stats, "neardup_filtering_after")
            record.write(writer)
    return stats

def toy_pipeline(input_path: pathlib.Path, output_dir: pathlib.Path):
    output_path = output_dir / "filtered_records.warc.wet.gz"
    stats_json = output_dir / "filtering_stats.json"
    print("Filtering records...")
    stats = filter_records(input_path, output_path)
    with stats_json.open("w") as json_out:
        json.dump(stats, json_out)
    
    for key, val in stats.items():
        if not isinstance(val, list):
            print(f"{key}: {val}")
    
    print("Filtering dup lines...")
    filtered_dup_lines = output_dir / "filtered_dup_lines.warc.wet.gz"
    exact_dupes_stats = remove_dup_lines([output_path], filtered_dup_lines)
    stats.update(exact_dupes_stats)

    print("Filtering neardups...")
    jaccard_filtered = output_dir / "filtered_jaccard.warc.wet.gz"
    neardup_stats = remove_neardups([filtered_dup_lines], jaccard_filtered)
    
    stats.update(neardup_stats)

    for key, val in stats.items():
        if not isinstance(val, list):
            print(f"{key}: {val}")
    
def merge_stats(stats_to_update, stats_to_add):
    for key, val in stats_to_add.items():
        if isinstance(val, list):
            elements = stats_to_update.get(key, [])
            stats_to_update[key] = elements + val
        else:
            count = stats_to_update.get(key, 0)
            stats_to_update[key] = count + val

def url_pipeline(output_dir: pathlib.Path, max_urls: int):
    wet_urls = get_wet_file_urls()[:max_urls]
    download_dir = output_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    def process_single_wet_file(wet_url: str, output_dir: os.PathLike) -> dict:
        wet_path = output_dir / pathlib.Path(wet_url).name
        wet_output_path = wet_path.with_suffix(".filtered.warc.wet.gz").name
        try:
            download_wet_file(wet_url, wet_path)
            stats = filter_records(wet_path, wet_output_path)
        finally:
            shutil.rmtree(wet_path)
        return stats, wet_output_path
        
    # Set up the executor
    num_cpus = len(os.sched_getaffinity(0))
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
    futures = []
    print(f"Filtering records with {num_cpus} CPUs...")
    filter_start_time = time.time()
    for i, wet_url in enumerate(wet_urls):
        print(f"Processing file {i+1}/{len(wet_urls)}: {wet_url}")
        future = executor.submit(process_single_wet_file, wet_url, download_dir)
        futures.append(future)

    # Collect results
    stats = {}
    output_paths = []
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(wet_urls)):
        url_stats, wet_output_path  = future.result()
        merge_stats(stats, url_stats)
        output_paths.append(wet_output_path)

    for key, val in stats.items():
        if not isinstance(val, list):
            print(f"{key}: {val}")

    print("Filtering dup lines...")
    line_filter_start_time = time.time()
    stats["filtering_time_seconds"] = line_filter_start_time - filter_start_time

    filtered_dup_lines = output_dir / "filtered_dup_lines.warc.wet.gz"
    exact_dupes_stats = remove_dup_lines(output_paths, filtered_dup_lines)
    stats.update(exact_dupes_stats)

    print("Filtering neardups...")
    neardup_filter_start_time = time.time()
    stats["line_filtering_time_seconds"] = neardup_filter_start_time - filter_start_time
    jaccard_filtered = output_dir / "filtered_jaccard.warc.wet.gz"
    neardup_stats = remove_neardups([filtered_dup_lines], jaccard_filtered)
    
    stats.update(neardup_stats)
    end_time = time.time()
    stats["neardup_filter_time_seconds"] = end_time - filter_start_time

if __name__ == "__main__":
    args = parser.parse_args()
    input_path = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output_dir)
    if args.input_path == "URLS":
        url_pipeline(output_dir, args.max_urls)
    else:
        toy_pipeline(input_path, output_dir)
        

    