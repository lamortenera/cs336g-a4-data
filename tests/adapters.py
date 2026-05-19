from __future__ import annotations

import os
from typing import Any
from cs336_data import assignment



def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return assignment.extract_text(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return assignment.identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return assignment.mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return assignment.mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return assignment.mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return assignment.classify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return assignment.classify_hate_speech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    return assignment.gopher_quality_filter(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    return assignment.exact_line_deuplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    return assignment.minhash_deduplication(input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory)
