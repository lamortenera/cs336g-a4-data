import os

import resiliparse
import resiliparse.parse.encoding as resiliparse_encoding
import resiliparse.extract.html2text as resiliparse_html2text
import fasttext
import pathlib
import re
import nltk
import mmh3
from collections import Counter, defaultdict
from collections.abc import Callable, Generator


_DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"
_LANGUAGE_MODEL_PATH = _DATA_PATH / "lid.176.bin"
_NSF_MODEL_PATH = _DATA_PATH / "jigsaw_fasttext_bigrams_nsfw_final.bin"
_HATE_SPEECH_MODEL_PATH = _DATA_PATH / "jigsaw_fasttext_bigrams_hatespeech_final.bin"
_QUALITY_MODEL_PATH = _DATA_PATH / "quality_model_large.bin"

def identify_language(text: str, normalized=False):
    if not hasattr(identify_language, "model"):
        identify_language.model = fasttext.FastText.load_model(_LANGUAGE_MODEL_PATH.as_posix())
    if not normalized:
        text = text.replace("\n", " ")
    labels, scores = identify_language.model.predict(text)
    best_label = labels[0].replace("__label__", "")
    return best_label, scores[0]
    
def extract_text(html_bytes: bytes) -> str:
    """Extract text from HTML bytes.

    Args:
        html_bytes: The HTML content as bytes.
    """
    encoding = resiliparse_encoding.detect_encoding(html_bytes)
    html_string = html_bytes.decode(encoding)
    return resiliparse_html2text.extract_plain_text(html_string)

def mask_emails(text: str):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.subn(email_pattern, '|||EMAIL_ADDRESS|||', text)

def mask_phone_numbers(text: str):
    phone_pattern = r'(?<![\d.-a-zA-Z])(?:(?:\+|00)\d{1,3}[-. ]?)?\(?[0-9]{2,4}\)?[-. ]?[0-9]{2,4}[-. ]?[0-9]{3,5}(?![\d.-a-zA-Z])'
    return re.subn(phone_pattern, '|||PHONE_NUMBER|||', text)

def mask_ips(text: str):
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    return re.subn(ip_pattern, '|||IP_ADDRESS|||', text)

def classify_nsfw(text: str, normalized=False):
    if not hasattr(classify_nsfw, "model"):
        classify_nsfw.model = fasttext.FastText.load_model(_NSF_MODEL_PATH.as_posix())
    if not normalized:
        text = text.replace("\n", " ")
    labels, scores = classify_nsfw.model.predict(text)
    best_label = labels[0].replace("__label__", "")
    return best_label, scores[0]

def classify_hate_speech(text: str, normalized=False):
    if not hasattr(classify_hate_speech, "model"):
        classify_hate_speech.model = fasttext.FastText.load_model(_HATE_SPEECH_MODEL_PATH.as_posix())
    if not normalized:
        text = text.replace("\n", " ")
    labels, scores = classify_hate_speech.model.predict(text)
    best_label = labels[0].replace("__label__", "")
    return best_label, scores[0]

def gopher_quality_filter(text: str) -> bool:
    """Return false if the text satisfies any of these conditions:

    • Contain less than 50 or more than 100,000 words.
    • Have a mean word length outside the range of 3 to 10 characters.
    • Have more than 30% of lines ending with an ellipsis (“...”).
    • Contain less than 80% of words with at least one alphabetic character.
    """
    words = nltk.word_tokenize(text, preserve_line=True)
    if len(words) < 50 or len(words) > 100000:
        return False
    mean_word_length = sum(len(word) for word in words) / len(words)
    if mean_word_length < 3 or mean_word_length > 10:
        return False
    lines = text.splitlines()
    if len(lines) > 0 and sum(line.endswith("...") for line in lines) / len(lines) > 0.3:
        return False
    if sum(any(c.isalpha() for c in word) for word in words) / len(words) < 0.8:
        return False
    return True 

def classify_quality(text: str, normalized=False):
    if not hasattr(classify_quality, "model"):
        classify_quality.model = fasttext.FastText.load_model(_QUALITY_MODEL_PATH.as_posix())
    if not normalized:
        text = text.replace("\n", " ")
    labels, scores = classify_quality.model.predict(text)
    is_good = labels[0] == "__label__positive"
    return is_good, scores[0]

def exact_line_deduplication(
        input_files: list[os.PathLike], 
        output_directory: os.PathLike):
    counter = Counter()
    for input_file in input_files:
        with input_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                key = mmh3.hash128(line)
                counter[key] += 1
    for input_file in input_files:
        output_file = output_directory / input_file.name
        with input_file.open("r", encoding="utf-8", errors="ignore") as f_in, \
             output_file.open("w", encoding="utf-8") as f_out:
            for line in f_in:
                key = mmh3.hash128(line)
                if counter[key] == 1:
                    f_out.write(line)

def exact_line_dedup_records(
        input_records: Callable[[], Generator[tuple[object, str], None, None]]) -> Generator[tuple[object, str], None, None]:
    counter = Counter()
    for _, text in input_records():
        for line in text.splitlines():
            key = mmh3.hash128(line)
            counter[key] += 1

    for record, text in input_records():
        filtered_lines = []
        for line in text.splitlines():
            key = mmh3.hash128(line)
            if counter[key] == 1:
                filtered_lines.append(line)
        filtered_text = "\n".join(filtered_lines)
        yield record, filtered_text

def records_from_files(input_files: list[os.PathLike]) -> Generator[tuple[os.PathLike, str], None, None]:
    for input_file in input_files:
        with input_file.open("r", encoding="utf8", errors="ignore") as f:
            yield input_file, f.read()

def exact_line_dedup_files(
        input_files: list[os.PathLike], 
        output_directory: os.PathLike):
    input_records = lambda : records_from_files(input_files)
    for path, filtered_text in exact_line_dedup_records(input_records):
        output_file = output_directory / path.name
        with output_file.open("w", output_file) as f_out:
            f_out.write(filtered_text)

def get_minhash(ngrams: set[tuple[str, ...]], num_hashes: int) -> list[int]:
    result = []
    for i in range(num_hashes):
        min_hash = min(mmh3.hash(" ".join(ngram), seed=i) for ngram in ngrams)
        result.append(min_hash)
    return tuple(result)

def connected_components_rec(graph, id_to_cc, curr_id, curr_cc):
    if curr_id in id_to_cc:
        return
    id_to_cc[curr_id] = curr_cc
    for next_id in graph[curr_id]:
        connected_components_rec(graph, id_to_cc, next_id, curr_cc)
    
def connected_components(graph, id_to_cc):
    for i, id in enumerate(graph.keys()):
        connected_components_rec(graph, id_to_cc, id, i)

def minhash_deduplication(input_files: list[os.PathLike],
                          num_hashes: int,
                          num_bands: int,
                          ngram_len: int,
                          jaccard_threshold: float,
                          output_directory: os.PathLike):
        hashes = []
        for input_file in input_files:
            with input_file.open("r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                tokens = nltk.word_tokenize(content, preserve_line=True)
                ngrams = [tuple(tokens[i:i+ngram_len]) for i in range(len(tokens)-ngram_len+1)]
                hashes.append(get_minhash(ngrams, num_hashes))
        
        hash_tables = [defaultdict(list) for _ in range(num_bands)]
        band_length = num_hashes // num_bands

        for i, hash in enumerate(hashes):
            for b in range(num_bands):
                key = tuple(hash[b*band_length:(b+1)*band_length])
                hash_tables[b][key].append(i)
        
        graph = defaultdict(list)
        for table in hash_tables:
            for ids in table.values():
                if len(ids) <= 1:
                    continue
                for i, id1 in enumerate(ids):
                    hash1 = hashes[id1]
                    for _, id2 in enumerate(ids[i+1:]):
                        hash2 = hashes[id2]
                        sim = sum(h1 == h2 for h1, h2 in zip(hash1, hash2))/num_hashes
                        if sim > jaccard_threshold:
                            graph[id1].append(id2)
                            graph[id2].append(id1)

        id_to_cc = {}
        connected_components(graph, id_to_cc)

        cc_taken = set()
        for id, input_file in enumerate(input_files):
            cc = id_to_cc.get(id)
            if cc is not None:
                if cc not in cc_taken:
                    cc_taken.add(cc)
                else:
                    continue
            output_file = output_directory / input_file.name
            output_file.write_bytes(input_file.read_bytes())


def minhash_deduplication_from_records(
        input_records: Callable[[], Generator[tuple[object, str], None, None]],
        num_hashes: int,
        num_bands: int,
        ngram_len: int,
        jaccard_threshold: float) -> Generator[object, None, None]:
        hashes = []
        for _, content in input_records():
            tokens = nltk.word_tokenize(content, preserve_line=True)
            ngrams = [tuple(tokens[i:i+ngram_len]) for i in range(len(tokens)-ngram_len+1)]
            hashes.append(get_minhash(ngrams, num_hashes))
        
        hash_tables = [defaultdict(list) for _ in range(num_bands)]
        band_length = num_hashes // num_bands

        for i, hash in enumerate(hashes):
            for b in range(num_bands):
                key = tuple(hash[b*band_length:(b+1)*band_length])
                hash_tables[b][key].append(i)
        
        graph = defaultdict(list)
        for table in hash_tables:
            for ids in table.values():
                if len(ids) <= 1:
                    continue
                for i, id1 in enumerate(ids):
                    hash1 = hashes[id1]
                    for _, id2 in enumerate(ids[i+1:]):
                        hash2 = hashes[id2]
                        sim = sum(h1 == h2 for h1, h2 in zip(hash1, hash2))/num_hashes
                        if sim > jaccard_threshold:
                            graph[id1].append(id2)
                            graph[id2].append(id1)

        id_to_cc = {}
        connected_components(graph, id_to_cc)

        cc_taken = set()
        for id, (record_metadata, _) in enumerate(input_records):
            cc = id_to_cc.get(id)
            if cc is not None:
                if cc not in cc_taken:
                    cc_taken.add(cc)
                else:
                    continue
            yield record_metadata

def minhash_deduplication_from_files(input_files: list[os.PathLike],
                          num_hashes: int,
                          num_bands: int,
                          ngram_len: int,
                          jaccard_threshold: float,
                          output_directory: os.PathLike):
    input_records = lambda : records_from_files(input_files)
    for path in minhash_deduplication_from_records(
        input_records, num_hashes, num_bands, ngram_len, jaccard_threshold):
        output_file = output_directory / path.name
        output_file.write_bytes(path.read_bytes())