from fastwarc.warc import ArchiveIterator, WarcRecordType

import argparse

import assignment

parser = argparse.ArgumentParser()
parser.add_argument("input", help="the input WARC file")
parser.add_argument("n", help="the number of records to process", type=int)
parser.add_argument("--with_masks", help="whether to print out only examples where there was a mask", action="store_true")
parser.add_argument("--with_safety", help="whether to print out only examples where there was a safety filter hit", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    record_num = 0
    with open(args.input, "rb") as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            if record_num >= int(args.n):
                break
            content = record.reader.read()
            try:
                text = assignment.extract_text(content)
            except Exception as e:
                print(f"Error processing record {record_num}: {e}")
                continue
            if args.with_masks:
                masked_text, num_masks = assignment.mask_emails(text)
                masked_text, num_masks = assignment.mask_phone_numbers(masked_text)
                masked_text, num_masks = assignment.mask_ips(masked_text)
                if num_masks > 0:
                    print(f"Masked text (with {num_masks} masks):")
                    print(masked_text)
                    record_num += 1
            elif args.with_safety:
                nsfw_label, nsfw_confidence = assignment.classify_nsfw(text)
                hate_speech_label, hate_speech_confidence = assignment.classify_hate_speech(text)
                if nsfw_label == "nsfw" or hate_speech_label == "hatespeech":
                    print(f"Safety filter hit:")
                    print(f"NSFW: {nsfw_label} (confidence: {nsfw_confidence:.4f})")
                    print(f"Hate Speech: {hate_speech_label} (confidence: {hate_speech_confidence:.4f})")
                    print(text)
                    record_num += 1
            else:
                print(f"### Record {record_num}: ###\n")
                language, confidence = assignment.identify_language(text)
                print(f"Language: {language} (confidence: {confidence:.4f})\n")
                print(text)
                record_num += 1