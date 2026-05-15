import resiliparse
import resiliparse.parse.encoding as resiliparse_encoding
import resiliparse.extract.html2text as resiliparse_html2text
import fasttext
import pathlib
import re
import nltk

_DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"
_LANGUAGE_MODEL_PATH = _DATA_PATH / "lid.176.bin"
_NSF_MODEL_PATH = _DATA_PATH / "jigsaw_fasttext_bigrams_nsfw_final.bin"
_HATE_SPEECH_MODEL_PATH = _DATA_PATH / "jigsaw_fasttext_bigrams_hatespeech_final.bin"

def identify_language(text: str):
    model = fasttext.FastText.load_model(_LANGUAGE_MODEL_PATH.as_posix())
    labels, scores = model.predict(text.replace("\n", " "))
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

def classify_nsfw(text: str):
    model = fasttext.FastText.load_model(_NSF_MODEL_PATH.as_posix())
    labels, scores = model.predict(text.replace("\n", " "))
    best_label = labels[0].replace("__label__", "")
    return best_label, scores[0]

def classify_hate_speech(text: str):
    model = fasttext.FastText.load_model(_HATE_SPEECH_MODEL_PATH.as_posix())
    labels, scores = model.predict(text.replace("\n", " "))
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